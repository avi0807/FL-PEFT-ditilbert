# Federated Fine-Tuning of DistilBERT with LoRA

Binary sentiment classification on IMDb using parameter-efficient federated learning.

---

## Overview

This project implements federated fine-tuning of `distilbert-base-uncased` using Low-Rank Adaptation (LoRA) across multiple GPU clients coordinated by the [Flower](https://flower.ai) framework. Training data never leaves each client — only LoRA weight updates are transmitted to the server for aggregation via FedAvg.

The codebase reproduces the LoRA configuration comparison from Table II of the target paper, evaluating three adapter configurations across ten federated rounds with two physical GPU clients.

---

## Results

| Configuration | F1 Score | Runtime (s) | Memory |
|---------------|----------|-------------|--------|
| r = 4 (Q, V) | 0.9184 | 1,721 | Low |
| r = 8 (Q, V) | 0.9291 | 1,939 | Moderate |
| r = 8 (Q, K, V) | 0.9312 | 2,148 | High |

Evaluated on the full IMDb test set (25,000 samples) after 10 federated rounds, 3 local epochs per round, 2 clients.

---

## Project Structure

```
├── model.py          # DistilBERT + LoRA construction, train/eval utilities
├── client.py         # Flower NumPyClient — local training, data partitioning
├── server.py         # Flower server — FedAvg, global evaluation, CSV logging
├── utils.py          # Dataset loading, partitioning, metrics, seed management
├── inference.py      # Model loading and prediction logic for inference
├── api.py            # FastAPI server with /predict endpoints
├── index.html        # Web frontend (SentimentLens)
└── requirements.txt  # Pinned dependencies
```

---

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA 12.x (CPU also works, slower)
- WSL2 or Linux recommended for GPU support on Windows

Install dependencies:

```bash
pip install torch==2.10.0 transformers==5.1.0 peft==0.18.1 datasets==4.5.0 \
            flwr==1.26.1 numpy==2.2.6 tqdm==4.67.3 evaluate==0.4.2 \
            scikit-learn==1.4.2 fastapi uvicorn
```

---

## Training

### 1. Start the server

```bash
python server.py \
  --rounds 10 \
  --min-clients 2 \
  --min-fit-clients 2 \
  --min-eval-clients 2 \
  --lora-config r8_qv
```

Wait until you see `Flower ECE: gRPC server running` before starting clients.

### 2. Start each client in a separate terminal

```bash
# Client 0
python client.py --client-id 0 --num-clients 2 --lora-config r8_qv

# Client 1
python client.py --client-id 1 --num-clients 2 --lora-config r8_qv
```

For clients on different machines, pass the server's IP:

```bash
python client.py --client-id 1 --num-clients 2 \
  --server-address 192.168.1.10:8080 \
  --lora-config r8_qv
```

### 3. LoRA configurations

| Flag | Rank | Target Matrices |
|------|------|-----------------|
| `r4_qv` | 4 | Query, Value |
| `r8_qv` | 8 | Query, Value |
| `r8_qkv` | 8 | Query, Key, Value |

Run all three experiments sequentially to reproduce the paper table. Each produces its own results file: `results_r4_qv.csv`, `results_r8_qv.csv`, `results_r8_qkv.csv`.

---

## Inference API

Start the API server after training:

```bash
# With trained weights
python api.py --lora-config r8_qv --checkpoint best_model_r8_qv.pt

# Without weights (falls back to pretrained SST-2 model for demo)
python api.py --lora-config r8_qv
```

Open `http://localhost:8000` for the web UI or `http://localhost:8000/docs` for the Swagger API.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web frontend |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single text prediction |
| POST | `/predict/batch` | Batch prediction (up to 32) |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An absolutely brilliant film. One of the best I have seen."}'
```

```json
{
  "label": "POSITIVE",
  "confidence": 0.9823,
  "scores": { "POSITIVE": 0.9823, "NEGATIVE": 0.0177 },
  "text_length": 12
}
```

---

## Data Partitioning

The IMDb training set (25,000 samples) is split equally across clients using IID partitioning with `seed=42`. Each client holds 12,500 samples, of which 10% is reserved as local validation. Data never leaves the client.

To switch to non-IID partitioning, change `iid=True` to `iid=False` in `client.py`.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Batch size | 32 |
| Local epochs per round | 3 |
| LoRA alpha | 2 × rank |
| LoRA dropout | 0.1 |
| Max sequence length | 256 |
| Warmup ratio | 10% |

---

## Hardware Used

| Client | GPU | VRAM |
|--------|-----|------|
| Client 0 (server host) | RTX 3050 Laptop | 6 GB |
| Client 1 | RTX 4060 | 8 GB |

---

## Collecting Results

After all experiments complete, run:

```python
import csv

configs = ["r4_qv", "r8_qv", "r8_qkv"]
for cfg in configs:
    with open(f"results_{cfg}.csv") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    print(f"{cfg}  F1={last['f1']}  acc={last['accuracy']}  loss={last['loss']}")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Sampling failed: available clients < requested` | Add `--min-fit-clients 2 --min-eval-clients 2` to server command |
| Server hangs on final round | Set `fraction_evaluate=0.0` in `server.py` strategy |
| `Connection refused` on client | Server not started yet or wrong IP — check `--server-address` |
| CUDA out of memory | Reduce `BATCH_SIZE` in `client.py` |
| Client misses early rounds | Start both clients before the server begins Round 1 |
