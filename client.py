"""
client.py
---------
Flower federated-learning client.

Usage (one terminal per client):
    python client.py --client-id 0 --num-clients 2 --server-address 127.0.0.1:8080
    python client.py --client-id 1 --num-clients 2 --server-address 127.0.0.1:8080

For cross-machine setups replace 127.0.0.1 with the server's IP address.
"""

import argparse
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from model import (
    build_model,
    load_tokenizer,
    get_trainable_params,
    set_trainable_params,
    train_one_epoch,
    evaluate as eval_model,
    build_optimizer_and_scheduler,
    MAX_LENGTH,
) 
from utils import (
    set_seed,
    setup_logging,
    load_imdb,
    partition_dataset,
    tokenize_dataset,
    compute_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
LOCAL_EPOCHS  = 3
SEED          = 42

# ---------------------------------------------------------------------------
# LoRA configurations — defined at module level so main() can access them
# ---------------------------------------------------------------------------
LORA_CONFIGS = {
    "r4_qv":  {"lora_r": 4, "lora_target_modules": ["q_lin", "v_lin"]},
    "r8_qv":  {"lora_r": 8, "lora_target_modules": ["q_lin", "v_lin"]},
    "r8_qkv": {"lora_r": 8, "lora_target_modules": ["q_lin", "k_lin", "v_lin"]},
}


# ---------------------------------------------------------------------------
# IMDb Flower client
# ---------------------------------------------------------------------------
class IMDbClient(fl.client.NumPyClient):
    """
    A Flower NumPyClient that:
      1. Holds a local shard of the IMDb training set.
      2. Trains the LoRA-adapted DistilBERT locally for LOCAL_EPOCHS epochs.
      3. Returns updated LoRA weights to the server.
      4. Can evaluate the current global model on its local validation slice.
    """

    def __init__(
        self,
        client_id: int,
        num_clients: int,
        device: torch.device,
        lora_r: int,
        lora_target_modules: List[str],
    ):
        super().__init__()
        self.client_id   = client_id
        self.num_clients = num_clients
        self.device      = device

        # Different seed per client ensures different data shuffling
        set_seed(SEED + client_id)

        # ── Build model & tokenizer ────────────────────────────────────────
        logger.info("Client %d: building model …", client_id)
        self.tokenizer = load_tokenizer()
        self.model     = build_model(
            device,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
        )

        # ── Load and partition IMDb ────────────────────────────────────────
        logger.info("Client %d: loading dataset partition …", client_id)
        imdb      = load_imdb()
        raw_train = partition_dataset(
            imdb["train"],
            num_clients=num_clients,
            client_id=client_id,
            seed=SEED,
            iid=True,    # set False for non-IID experiments
        )

        # Reserve 10% of the client's shard as local validation
        split     = raw_train.train_test_split(test_size=0.1, seed=SEED)
        raw_train = split["train"]
        raw_val   = split["test"]

        logger.info(
            "Client %d: tokenising %d train / %d val samples …",
            client_id, len(raw_train), len(raw_val),
        )
        train_tok = tokenize_dataset(raw_train, self.tokenizer, MAX_LENGTH)
        val_tok   = tokenize_dataset(raw_val,   self.tokenizer, MAX_LENGTH)

        self.train_loader = DataLoader(
            train_tok,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,      # 0 workers avoids multiprocessing issues on WSL/Windows
            pin_memory=(device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_tok,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        self.num_train_samples = len(raw_train)
        self.num_val_samples   = len(raw_val)

        logger.info(
            "Client %d ready – %d train batches, %d val batches.",
            client_id,
            len(self.train_loader),
            len(self.val_loader),
        )

    # ── Flower API ─────────────────────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current trainable parameters as a list of NumPy arrays."""
        params = get_trainable_params(self.model)
        return [p.numpy() for p in params]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load aggregated parameters received from the server into the model."""
        tensors = [torch.from_numpy(np.copy(p)) for p in parameters]
        set_trainable_params(self.model, tensors)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global parameters, perform LOCAL_EPOCHS of local training,
        then return the updated parameters plus training metrics.
        """
        # 1. Synchronise with global model weights from server
        self.set_parameters(parameters)

        # 2. Build fresh optimiser + scheduler for this round
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model,
            self.train_loader,
            lr=LEARNING_RATE,
            num_epochs=LOCAL_EPOCHS,
        )

        # 3. Start timer and reset GPU memory stats before training
        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # 4. Local training loop
        total_loss    = 0.0
        total_samples = 0
        for epoch in range(LOCAL_EPOCHS):
            epoch_loss, n = train_one_epoch(
                self.model, self.train_loader, optimizer, scheduler, self.device
            )
            total_loss    += epoch_loss * n
            total_samples += n
            logger.info(
                "Client %d | Epoch %d/%d | loss=%.4f",
                self.client_id, epoch + 1, LOCAL_EPOCHS, epoch_loss,
            )

        # 5. Record runtime and peak GPU memory usage
        runtime = time.time() - start_time

        if self.device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            mem_mb = 0.0

        avg_loss = total_loss / max(total_samples, 1)
        logger.info(
            "Client %d | Runtime: %.1fs | Peak VRAM: %.1f MB",
            self.client_id, runtime, mem_mb,
        )

        # 6. Return updated weights + metadata to server
        updated_params = self.get_parameters(config={})
        metrics: Dict[str, Scalar] = {
            "train_loss":    float(avg_loss),
            "train_samples": float(self.num_train_samples),
            "runtime_sec":   float(runtime),
            "memory_mb":     float(mem_mb),
        }
        return updated_params, self.num_train_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on the client's local validation slice.
        Called by the server for distributed evaluation each round.
        """
        self.set_parameters(parameters)

        loss, preds, labels = eval_model(
            self.model, self.val_loader, self.device
        )
        metrics = compute_metrics(preds, labels, loss)

        return float(loss), self.num_val_samples, {
            "accuracy": metrics["accuracy"],
            "f1":       metrics["f1"],
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Flower federated-learning client for IMDb LoRA fine-tuning"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Zero-based client index (0, 1, 2, …)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Total number of clients participating in training (default: 2)",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="gRPC address of the Flower server (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default="r8_qv",
        choices=["r4_qv", "r8_qv", "r8_qkv"],
        help=(
            "LoRA configuration: "
            "r4_qv = rank 4 on Q+V, "
            "r8_qv = rank 8 on Q+V, "
            "r8_qkv = rank 8 on Q+K+V"
        ),
    )

    # ── IMPORTANT: parse args first, then use them ─────────────────────────
    args = parser.parse_args()

    # Resolve LoRA config from parsed args
    cfg = LORA_CONFIGS[args.lora_config]

    # Device selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    logger.info(
        "Client %d using device: %s | LoRA config: %s",
        args.client_id, device, args.lora_config,
    )

    # Build client
    client = IMDbClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        device=device,
        lora_r=cfg["lora_r"],
        lora_target_modules=cfg["lora_target_modules"],
    )

    # Connect to the Flower server and start the federated loop
    logger.info(
        "Client %d connecting to server at %s …",
        args.client_id,
        args.server_address,
    )
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()










