"""
inference.py
------------
Loads the trained LoRA-adapted DistilBERT model and exposes
a simple predict() function used by the FastAPI server.

The model is loaded ONCE at startup and reused for all requests.

"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional

from transformers import DistilBertTokenizerFast
from peft import PeftModel, PeftConfig

from model import (
    build_model,
    load_tokenizer,
    MODEL_NAME,
    MAX_LENGTH,
    NUM_LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals — loaded once at startup
# ---------------------------------------------------------------------------
_model     = None
_tokenizer = None
_device    = None

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_inference_model(
    lora_config: str = "r8_qv",
    checkpoint_path=None,
    device=None,
):
    global _model, _tokenizer, _device

    _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model on device: %s", _device)

    if checkpoint_path and Path(checkpoint_path).exists():
        # Use your trained LoRA weights if available
        LORA_CONFIGS = {
            "r4_qv":  {"lora_r": 4, "lora_target_modules": ["q_lin", "v_lin"]},
            "r8_qv":  {"lora_r": 8, "lora_target_modules": ["q_lin", "v_lin"]},
            "r8_qkv": {"lora_r": 8, "lora_target_modules": ["q_lin", "k_lin", "v_lin"]},
        }
        cfg    = LORA_CONFIGS[lora_config]
        _model = build_model(_device, lora_r=cfg["lora_r"], lora_target_modules=cfg["lora_target_modules"])
        state  = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(state, strict=False)
        logger.info("Loaded LoRA checkpoint from %s", checkpoint_path)

    else:
        # Fall back to HuggingFace pretrained SST-2 model
        # distilbert-base-uncased-finetuned-sst-2-english is trained on
        # sentiment classification — works great for IMDb demos
        logger.info("No checkpoint found — loading pretrained SST-2 model from HuggingFace …")
        from transformers import pipeline as hf_pipeline
        global _pipeline
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if _device.type == "cuda" else -1,
        )
        _model = None   # signal to predict() to use pipeline instead
        logger.info("Pretrained SST-2 model loaded.")

    _tokenizer = load_tokenizer()
    logger.info("Ready for inference.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(text: str) -> Dict:
    global _pipeline

    # Use HuggingFace pipeline if no LoRA model loaded
    if _model is None:
        result = _pipeline(text, truncation=True, max_length=512)[0]
        label  = result["label"]        # already "POSITIVE" or "NEGATIVE"
        conf   = round(result["score"], 4)
        other  = round(1 - conf, 4)
        return {
            "label":      label,
            "confidence": conf,
            "scores": {
                "POSITIVE": conf   if label == "POSITIVE" else other,
                "NEGATIVE": other  if label == "POSITIVE" else conf,
            },
        }

    # Use your trained LoRA model
    inputs = _tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    outputs = _model(
        input_ids=inputs["input_ids"].to(_device),
        attention_mask=inputs["attention_mask"].to(_device)
    )
    probs    = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    return {
        "label":      LABEL_MAP[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "scores": {
            "NEGATIVE": round(probs[0].item(), 4),
            "POSITIVE": round(probs[1].item(), 4),
        },
    }

@torch.no_grad()
def predict_batch(texts: list) -> list:
    """
    Run sentiment inference on a list of texts.
    More efficient than calling predict() in a loop.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_inference_model() first.")

    inputs = _tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids      = inputs["input_ids"].to(_device)
    attention_mask = inputs["attention_mask"].to(_device)

    outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
    probs   = torch.softmax(outputs.logits, dim=-1)   # shape: [batch, 2]

    results = []
    for i in range(len(texts)):
        p        = probs[i]
        pred_idx = p.argmax().item()
        results.append({
            "label":      LABEL_MAP[pred_idx],
            "confidence": round(p[pred_idx].item(), 4),
            "scores": {
                "NEGATIVE": round(p[0].item(), 4),
                "POSITIVE": round(p[1].item(), 4),
            },
        })
    return results

def get_model_info() -> Dict:
    """Return metadata about the loaded model."""
    if _model is None:
        return {"status": "not loaded"}

    total_params     = sum(p.numel() for p in _model.parameters())
    trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)

    return {
        "status":           "loaded",
        "base_model":       MODEL_NAME,
        "device":           str(_device),
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "trainable_pct":    round(100 * trainable_params / total_params, 4),
        "max_length":       MAX_LENGTH,
        "num_labels":       NUM_LABELS,
        "labels":           LABEL_MAP,
    }