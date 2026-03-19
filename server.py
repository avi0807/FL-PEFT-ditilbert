"""
server.py
---------
Flower federated-learning server.

Usage:
    python server.py --rounds 10 --min-clients 2 --min-fit-clients 2 --min-eval-clients 2 --lora-config r8_qv
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from model import (
    build_model,
    load_tokenizer,
    get_trainable_params,
    set_trainable_params,
    evaluate as eval_model,
    MAX_LENGTH,
)
from utils import (
    set_seed,
    setup_logging,
    load_imdb,
    tokenize_dataset,
    compute_metrics,
    log_round_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global objects - initialised once in main(), shared via closure
# ---------------------------------------------------------------------------
_device:           Optional[torch.device]    = None
_model:            Optional[torch.nn.Module] = None
_test_loader:      Optional[DataLoader]      = None
_lora_config_name: str                       = "r8_qv"  # updated in main()

BATCH_SIZE = 48
SEED       = 42


# ---------------------------------------------------------------------------
# Global evaluation function (called by Flower after each aggregation round)
# ---------------------------------------------------------------------------

def evaluate_global(
    server_round: int,
    parameters,                    # NDArrays in flwr 1.10+, plain list of ndarrays
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """
    Flower calls this after each aggregation round.
    Loads aggregated parameters into the global model and
    evaluates on the full IMDb test set.
    """
    if _model is None or _test_loader is None:
        logger.warning("Global model or test loader not initialised; skipping eval.")
        return None

    # Apply aggregated weights (parameters is already a plain list of ndarrays in flwr 1.10+)
    tensors = [torch.from_numpy(np.copy(p)) for p in parameters]
    set_trainable_params(_model, tensors)

    # Evaluate on full test set
    loss, preds, labels = eval_model(_model, _test_loader, _device)
    metrics             = compute_metrics(preds, labels, loss)

    # Save to config-specific CSV using the global _lora_config_name
    log_round_metrics(
        metrics,
        round_num=server_round,
        output_path=f"results_{_lora_config_name}.csv",
    )

    logger.info(
        "[Server] Round %d - test accuracy=%.4f  f1=%.4f  loss=%.6f",
        server_round,
        metrics["accuracy"],
        metrics["f1"],
        metrics["loss"],
    )
    return float(loss), {
        "accuracy": metrics["accuracy"],
        "f1":       metrics["f1"],
    }


# ---------------------------------------------------------------------------
# Custom FedAvg strategy with fit-metrics logging
# ---------------------------------------------------------------------------

class FedAvgWithLogging(FedAvg):
    """Extends FedAvg to log weighted-average train loss after each round."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            weighted_loss = sum(
                fit_res.metrics.get("train_loss", 0.0) * fit_res.num_examples
                for _, fit_res in results
            ) / max(total_samples, 1)
            logger.info(
                "[Server] Round %d - aggregated train loss (weighted avg): %.4f",
                server_round,
                weighted_loss,
            )

        return aggregated_params, aggregated_metrics


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_average_fit_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    total_samples = sum(n for n, _ in metrics)
    aggregated: Dict[str, float] = {}
    for key in metrics[0][1].keys():
        aggregated[key] = sum(
            n * float(m.get(key, 0.0)) for n, m in metrics
        ) / max(total_samples, 1)
    return aggregated


def _weighted_average_eval_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    total_samples = sum(n for n, _ in metrics)
    aggregated: Dict[str, float] = {}
    for key in metrics[0][1].keys():
        aggregate