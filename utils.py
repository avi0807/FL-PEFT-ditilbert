"""
utils.py
--------
Shared utilities:
  - Reproducibility (seed setting)
  - Dataset loading & partitioning helpers
  - Tokenisation helper
  - Metric computation (accuracy, F1, loss)
  - Results logging to CSV
"""

import os
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
import evaluate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For GPU reproducibility (no-op on CPU-only machines)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_imdb(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Download / load the IMDb dataset from HuggingFace Hub.

    Returns a DatasetDict with keys 'train' and 'test'.
    The 'unsupervised' split is dropped as it has no labels.
    """
    logger.info("Loading IMDb dataset …")
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    # Keep only train / test
    dataset = DatasetDict(
        train=dataset["train"],
        test=dataset["test"],
    )
    logger.info(
        "IMDb loaded – train: %d, test: %d",
        len(dataset["train"]),
        len(dataset["test"]),
    )
    return dataset


# ---------------------------------------------------------------------------
# Dataset partitioning
# ---------------------------------------------------------------------------

def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
    iid: bool = True,
) -> Dataset:
    """
    Partition `dataset` into `num_clients` non-overlapping shards
    and return the shard that belongs to `client_id` (0-indexed).

    Parameters
    ----------
    dataset    : HuggingFace Dataset (train split)
    num_clients: total number of federated clients
    client_id  : index of the current client (0 … num_clients-1)
    seed       : RNG seed for reproducible shuffling
    iid        : if True, shuffle first (IID); if False, keep sorted by label
                 (non-IID / label-skewed partitioning)
    """
    assert 0 <= client_id < num_clients, (
        f"client_id {client_id} out of range [0, {num_clients})"
    )

    indices = list(range(len(dataset)))

    if iid:
        # Shuffle so each client gets a random mix of labels
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        # Sort by label to create non-IID shards (label skew)
        labels  = dataset["label"]
        indices = sorted(indices, key=lambda i: labels[i])

    # Divide indices into num_clients equal-ish chunks
    chunk_size = len(indices) // num_clients
    start = client_id * chunk_size
    # Last client gets any remainder
    end   = start + chunk_size if client_id < num_clients - 1 else len(indices)

    client_indices = indices[start:end]
    partition = dataset.select(client_indices)

    logger.info(
        "Client %d/%d – partition size: %d samples",
        client_id,
        num_clients,
        len(partition),
    )
    return partition


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 256,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 256,
) -> Dataset:
    """
    Tokenise all examples in `dataset` and return a new Dataset
    with columns: input_ids, attention_mask, labels.
    """

    def _tokenize(batch):
        encoded = tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = batch[label_column]
        return encoded

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenising",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

# Load HuggingFace `evaluate` metrics once
_accuracy_metric = evaluate.load("accuracy")
_f1_metric       = evaluate.load("f1")


def compute_metrics(
    predictions: List[int],
    references: List[int],
    loss: float,
) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, and include the given loss.

    Returns a dict with keys: accuracy, f1, loss.
    """
    accuracy = _accuracy_metric.compute(
        predictions=predictions, references=references
    )["accuracy"]

    f1 = _f1_metric.compute(
        predictions=predictions,
        references=references,
        average="macro",
    )["f1"]

    return {
        "accuracy": round(float(accuracy), 4),
        "f1":       round(float(f1), 4),
        "loss":     round(float(loss), 6),
    }


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def log_round_metrics(
    metrics: Dict[str, float],
    round_num: int,
    output_path: str = "results.csv",
) -> None:
    """
    Append per-round metrics to a CSV file suitable for publication tables.
    Creates the file with a header row if it doesn't exist yet.
    """
    path = Path(output_path)
    write_header = not path.exists()

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "accuracy", "f1", "loss"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({"round": round_num, **metrics})

    logger.info(
        "Round %d  accuracy=%.4f  f1=%.4f  loss=%.6f",
        round_num,
        metrics["accuracy"],
        metrics["f1"],
        metrics["loss"],
    )


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to print timestamped messages to stdout."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )