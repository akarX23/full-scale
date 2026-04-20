# Edge and Cloud Systems Course Project

A unified workflow for CNN model training/reduction and heterogeneous inference serving using OpenVINO.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Repository Layout](#2-repository-layout)
- [3. Prerequisites](#3-prerequisites)
- [4. Local Setup](#4-local-setup)
- [5. Training Workflow](#5-training-workflow)
- [6. Inference Server Workflow](#6-inference-server-workflow)
- [7. Deployment Options](#7-deployment-options)
- [8. Benchmarking](#8-benchmarking)
- [9. Troubleshooting](#9-troubleshooting)

## 1. Project Overview

This project supports:

- Training CNN models (LeNet, AlexNet, VGG16, ResNet18).
- Model reduction through pruning and/or quantization.
- Running a FastAPI-based inference server with OpenVINO on available devices (CPU/GPU/NPU).
- Benchmarking throughput and latency across devices and batch sizes.

## 2. Repository Layout

- `harness/train/`: model training and reduction scripts.
- `harness/benchmark/`: benchmarking and analysis tools.
- `inference_server/`: FastAPI inference server and scheduling pipeline.
- `models/`: saved model weights and OpenVINO artifacts.
- `data/`: datasets and generated training statistics.
- `docs/`: report and supporting documentation.

## 3. Prerequisites

- Python 3.10 or newer (3.11 recommended).
- `pip` package manager.
- Git (for local repository initialization and version control).
- OpenVINO-compatible drivers/runtime for the target hardware.

## 4. Local Setup

### 4.1 Clone repository

```bash
git clone https://github.com/akarX23/full-scale.git
cd full-scale
```

### 4.2 Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install dependencies

Windows (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS (bash):

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Training Workflow

### 5.1 Train dense model + reduce model

Run from project root:

```powershell
python -m harness.train.train_cnn --model lenet --batch-size 64 --train-dense --train-epochs 30 --reduction-mode both --device GPU
```

Key options:

- `--model`: `lenet | alexnet | vgg16 | resnet18`
- `--batch-size`: training/eval batch size.
- `--train-dense`: train dense model from scratch first.
- `--reduction-mode`: `prune | quantize | both`.
- `--device`: preferred training device (`GPU`, `XPU`, `CPU`).
- `--save-dir`: output root for model artifacts (default: `./models`).

### 5.2 Output artifacts

Expected outputs are written under:

- Dense Model: `models/<model_name>/model.pth`
- Pruned Model: `models/<model_name>/pruned/`
- Quantized Model: `models/<model_name>/quantized/`

## 6. Inference Server Workflow

### 6.1 Run server (local)

```powershell
python -m inference_server --served-model lenet --model-type dense --host 0.0.0.0 --port 8000
```

Defaults (if omitted):

- Host: `0.0.0.0`
- Port: `8000`
- Served model: `lenet`
- Model type: `dense`

### 6.2 Environment variables

You can configure deployment using environment variables:

- `SERVED_MODEL`: `lenet | alexnet | resnet18 | vgg16`
- `MODEL_TYPE`: `dense | pruned | quantized`
- `MODEL_DIR`: model artifact root (default `./models`)
- `DATA_DIR`: data root (default `./data`)
- `HOST`, `PORT`
- `IGNORE_DEVICES`: comma-separated list (example: `GPU,NPU`)

### 6.3 Health check

After server startup:

- Swagger UI: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/health`

## 7. Deployment Options

### Option A: Docker (single container)

Build image:

```powershell
docker build -t ecs-inference-server:latest .
```

Run container:

```powershell
docker run --rm -p 8000:8000 `
  -e SERVED_MODEL=lenet `
  -e MODEL_TYPE=dense `
  -v ${pwd}/models:/app/models `
  -v ${pwd}/data:/app/data `
  -v ${pwd}/logs:/app/logs `
  ecs-inference-server:latest
```

### Option B: Docker Compose (additional deployable option)

Start:

```powershell
docker compose up --build -d
```

Stop:

```powershell
docker compose down
```

Update the environment values in `docker-compose.yml` to switch model type/device behavior.

## 8. Benchmarking

### 8.1 Benchmark live inference server

```powershell
python -m harness.benchmark.bench_server --total-requests 200 --batch-size 1 --max-concurrency 16 --base-url http://127.0.0.1:8000
```

### 8.2 Benchmark preprocessed path

```powershell
python -m harness.benchmark.bench_preprocessed --total-requests 200 --batch-size 1 --max-concurrency 16 --base-url http://127.0.0.1:8000
```

## 9. Troubleshooting

- If `git` is not found, install Git and restart terminal.
- If OpenVINO cannot detect devices, check host drivers and runtime installation.
- If upload endpoints fail with form-data errors, ensure `python-multipart` is installed.
- If model loading fails, verify the model path under `models/<served_model>/<model_type>/`.
