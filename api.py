"""
api.py
------
FastAPI server exposing the trained DistilBERT + LoRA sentiment model.

Endpoints:
    GET  /              → serve the frontend UI (index.html)
    GET  /health        → health check
    GET  /model/info    → model metadata
    POST /predict       → single text prediction
    POST /predict/batch → batch text prediction

Usage:
    python api.py --lora-config r8_qv --checkpoint path/to/weights.pt
    
    Then open http://localhost:8000 in your browser.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import load_inference_model, predict, predict_batch, get_model_info
from utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Federated LoRA fine-tuned DistilBERT for sentiment classification",
    version="1.0.0",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        example="This movie was absolutely fantastic! A masterpiece of storytelling.",
    )


class PredictResponse(BaseModel):
    label:      str
    confidence: float
    scores:     dict
    text_length: int


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=32,
        example=["Great movie!", "Terrible film, waste of time."],
    )


class BatchPredictResponse(BaseModel):
    results: List[dict]
    count:   int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend UI."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    info = get_model_info()
    return {
        "status": "ok",
        "model_loaded": info["status"] == "loaded",
        "device": info.get("device", "unknown"),
    }


@app.get("/model/info")
async def model_info():
    """Return metadata about the loaded model."""
    return get_model_info()


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """
    Predict sentiment for a single text.

    Returns label (POSITIVE/NEGATIVE), confidence score,
    and raw softmax scores for both classes.
    """
    try:
        result = predict(request.text)
        result["text_length"] = len(request.text.split())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_sentiment(request: BatchPredictRequest):
    """
    Predict sentiment for a batch of texts (max 32).

    More efficient than calling /predict in a loop.
    """
    try:
        results = predict_batch(request.texts)
        return {"results": results, "count": len(results)}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="IMDb Sentiment Analysis API Server")
    parser.add_argument(
        "--lora-config",
        type=str,
        default="r8_qv",
        choices=["r4_qv", "r8_qv", "r8_qkv"],
        help="LoRA configuration used during training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved model weights (.pt file). If not provided, uses uninitialised weights.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU inference",
    )
    args = parser.parse_args()

    # Select device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    logger.info("API server using device: %s", device)

    # Load model before starting server
    load_inference_model(
        lora_config=args.lora_config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    logger.info("Starting API server on http://%s:%d", args.host, args.port)
    logger.info("Frontend UI available at http://localhost:%d", args.port)
    logger.info("API docs available at http://localhost:%d/docs", args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()