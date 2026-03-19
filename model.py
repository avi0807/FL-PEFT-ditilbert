from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2                      # binary sentiment: 0=negative, 1=positive
MAX_LENGTH = 256                     # token sequence length
# Target only the attention projection layers in DistilBERT


# ---------------------------------------------------------------------------
# Tokenizer (loaded once per process)
# ---------------------------------------------------------------------------
def load_tokenizer() -> DistilBertTokenizerFast:
    """Load and return the DistilBERT fast tokenizer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer loaded: %s", MODEL_NAME)
    return tokenizer


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(device, lora_r=8, lora_target_modules=None):
    """
    Build DistilBERT + LoRA with configurable rank and target modules.
    
    lora_r              : LoRA rank (4 or 8)
    lora_target_modules : list of attention layers to adapt
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_lin", "v_lin"]  # default

    base_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_r * 2,      # alpha = 2*r is standard practice
        lora_dropout=0.1,
        target_modules=lora_target_modules,
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    return model

# ---------------------------------------------------------------------------
# Parameter helpers (used by Flower client/server)
# ---------------------------------------------------------------------------
def get_trainable_params(model: nn.Module) -> List[torch.Tensor]:
    """Return a list of *trainable* parameter tensors (LoRA + head weights)."""
    return [p.detach().cpu().clone() for p in model.parameters() if p.requires_grad]


def set_trainable_params(model: nn.Module, params: List[torch.Tensor]) -> None:
    """
    Overwrite the trainable parameters of `model` with `params`.
    The order must match `get_trainable_params`.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(params), (
        f"Parameter count mismatch: model has {len(trainable)}, "
        f"but received {len(params)}"
    )
    with torch.no_grad():
        for p_model, p_new in zip(trainable, params):
            p_model.copy_(p_new.to(p_model.device))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Run one full pass over `dataloader`, updating model weights.

    Returns
    -------
    avg_loss : float
        Mean cross-entropy loss over all batches.
    num_samples : int
        Total number of training examples seen.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    progress_bar=tqdm(dataloader, desc="Training", leave=False)

    for batch in dataloader:
        # Move batch tensors to the target device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        total_loss  += loss.item() * labels.size(0)
        num_samples += labels.size(0)

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}",
             "avg_loss": f"{(total_loss / max(num_samples, 1)):.4f}"
             })


    avg_loss = total_loss / max(num_samples, 1)
    return avg_loss, num_samples


# ---------------------------------------------------------------------------
# Evaluation step
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate model on `dataloader`.

    Returns
    -------
    avg_loss : float
    all_preds : list[int]   – predicted class indices
    all_labels : list[int]  – ground-truth labels
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss  += outputs.loss.item() * labels.size(0)
        num_samples += labels.size(0)

        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(num_samples, 1)
    return avg_loss, all_preds, all_labels


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory
# ---------------------------------------------------------------------------
def build_optimizer_and_scheduler(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float = 2e-4,
    num_epochs: int = 1,
    warmup_ratio: float = 0.1,
) -> Tuple[torch.optim.Optimizer, object]:
    """
    Create AdamW optimizer and a linear warm-up scheduler
    configured for `num_epochs` passes over `dataloader`.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps   = len(dataloader) * num_epochs
    warmup_steps  = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler