# Developer Onboarding Guide

Welcome to the Universal Translation System! This guide helps you get set up and understand the system.

## Table of Contents
1. System Overview
2. Getting Started
3. Development Environment Setup
4. Key Components
5. Common Workflows
6. Troubleshooting
7. Resources

## System Overview
- **Edge (Client)**: Lightweight universal encoder (35MB base + 2–4MB vocabulary packs)
- **Cloud**: Decoder infrastructure returns translations
- **Coordinator**: Routes requests to least-loaded healthy decoders; shares pool via Redis with periodic mirroring to disk

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Docker + Docker Compose (for full local stack)
- CUDA GPU (optional for training; required for decoder GPU images)
- NVIDIA Container Toolkit (for GPU with Docker)

### Quick Start Paths
- CPU-only demo (fastest): run a single translation from CLI
- Full local stack (Docker): encoder + decoder + coordinator + redis + monitoring
- Train models (for production): run GPU training and export artifacts
- Production deployment: deploy on Kubernetes with secrets, persistence, and monitoring

### CPU-only Demo (no Docker)
```bash
# Create and activate a virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate setup (creates missing dirs and checks dependencies)
python main.py --mode setup --validate-only --verbose

# Run a quick translation
python main.py --mode translate --text "Hello world" --source-lang en --target-lang es

# Optional: smoke test utilities
python scripts/smoke_cpu.py
```

### Full Local Stack (Docker Compose)

#### 1) Configure environment (CRITICAL)
```bash
cp .env.example .env
# Edit .env and set (must be set before running services):
# - DECODER_JWT_SECRET or DECODER_JWT_SECRET_FILE
# - COORDINATOR_SECRET or COORDINATOR_SECRET_FILE
# - COORDINATOR_JWT_SECRET or COORDINATOR_JWT_SECRET_FILE
# - COORDINATOR_TOKEN or COORDINATOR_TOKEN_FILE
# - INTERNAL_SERVICE_TOKEN or INTERNAL_SERVICE_TOKEN_FILE
# - (Optional) JWT_PRIVATE_KEY_FILE and JWT_PUBLIC_KEY_PATH for RS256
# - HF_HUB_REPO_ID (and HF_TOKEN if private)
# - (Optional) LOG_LEVEL=INFO (DEBUG/INFO/WARNING/ERROR)
```
- Secrets can be provided via `*_FILE` envs in Docker/Kubernetes; the services read from files when present.

#### 2) Prefetch artifacts (required before starting services)
```bash
# Fetch production encoder, vocab packs, and adapters from your HF repo
python tools/prefetch_artifacts.py \
  --repo_id your-org/universal-translation-system \
  --packs latin cjk \
  --adapters es fr \
  --models production/encoder.onnx
```

#### 3) Start services
```bash
docker compose up -d --build encoder decoder redis coordinator prometheus grafana
# Encoder:     http://localhost:8000
# Decoder:     http://localhost:8001
# Coordinator: http://localhost:8002
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000
```

#### 4) Validate services
- Open in browser or use any HTTP client:
  - http://localhost:8000/health
  - http://localhost:8001/health
  - http://localhost:8002/health

#### GPU notes
- Ensure NVIDIA drivers + NVIDIA Container Toolkit are installed for decoder GPU support.
- Verify with:
```bash
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Training for Production
```bash
# Dynamic config (recommended; auto GPU detection)
python main.py --mode train

# Force N GPUs or distributed
python main.py --mode train --gpus 2 --distributed

# Use a YAML config
python main.py --mode train --training-config config/base.yaml

# Evaluate a checkpoint
python main.py --mode evaluate --checkpoint checkpoints/.../best_model.pt

# Export for deployment
python main.py --mode export --format onnx --output-dir models/export
# or
python main.py --mode export --format torchscript --output-dir models/export
```
- After exporting, either copy artifacts to `models/production/` and `vocabs/`, or publish to HF Hub and use `tools/prefetch_artifacts.py` with your repo.

### Production Deployment Checklist
- Secrets: set strong values for `DECODER_JWT_SECRET`, `COORDINATOR_JWT_SECRET`, `COORDINATOR_TOKEN`, `INTERNAL_SERVICE_TOKEN` (K8s Secrets, vault, etc.).
- Artifacts: publish models/vocabs/adapters to HF Hub or your store; ensure access tokens are configured.
- GPU: GPU nodes provisioned; requests/limits set; NVIDIA runtime installed.
- Persistence: volumes/PVCs for `models/`, `vocabs/`, Redis data.
- Networking: TLS termination, auth on public endpoints, proper CORS.
- Monitoring: Prometheus + Grafana dashboards and alerts.
- Scaling: multiple decoders; coordinator using Redis pool with mirroring to `configs/decoder_pool.json`.
- See: `docs/DEPLOYMENT.md`, `docs/DECODER_POOL.md`, `kubernetes/*.yaml`.

### Common Pitfalls
- Docker Compose env: do not rely on shell expansions like `$(openssl ...)` in compose; use `.env` variables (already supported).
- GPU on Windows: prefer WSL2 + NVIDIA toolkit.
- HF Hub: private repos require `HF_TOKEN` available to services.
- Missing artifacts: always prefetch models/vocabs before `docker compose up` or ensure decoder fetches on boot.

## Development Environment Setup

### Python
```bash
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Recommended VS Code extensions
- Python, Pylance, Black Formatter, Mermaid Preview, Docker

### Docker (optional)
```bash
# Encoder
docker build -t universal-encoder -f docker/encoder.Dockerfile .
docker run -p 8000:8000 universal-encoder

# Decoder (GPU required)
docker build -t universal-decoder -f docker/decoder.Dockerfile .
docker run --gpus all -p 8001:8001 universal-decoder
```

## Key Components
- `encoder/`: Python encoder (training-time)
- `encoder_core/`: C++ encoder core (deployment)
- `vocabulary/`: Vocabulary management (unified_vocabulary_creator.py, evolve_vocabulary.py)
- `cloud_decoder/`: Server-side decoder
- `coordinator/`: Advanced routing, health, metrics, Redis-backed pool with disk mirroring
- `monitoring/`: Prometheus/Grafana configs and collectors
- SDKs: `android/`, `ios/`, `flutter/universal_translation_sdk/`, `react-native/UniversalTranslationSDK/`, `web/universal-translation-sdk/`

## Common Workflows

### 1) Add a new language
1. Prepare data
2. Update/create vocabulary:
   ```bash
   python vocabulary/unified_vocabulary_creator.py
   # or
   python vocabulary/evolve_vocabulary.py
   ```
3. Train with the unified launcher:
   ```bash
   # YAML config
   python -m training.launch train --config config/archived_gpu_configs/training_generic_gpu.yaml
   
   # or dynamic (no YAML)
   python -m training.launch train --config dynamic --dynamic
   ```
   - Deprecated: `training/train_universal_system.py` and `training/distributed_train.py` (use the launcher instead)

### 2) Modify Encoder
1. Change code in `encoder/` or C++ in `encoder_core/`
2. Tests:
   ```bash
   pytest tests/test_encoder.py
   ```
3. Build C++ core if needed:
   ```bash
   cd encoder_core && mkdir build && cd build
   cmake .. && make
   ```

### 3) Modify Decoder
1. Change code in `cloud_decoder/`
2. Tests:
   ```bash
   pytest tests/test_decoder.py
   ```
3. (Optional) Benchmark locally or deploy via Docker

### 4) Training/Evaluation/Profile
- Train:
  ```bash
  python -m training.launch train --config config/base.yaml
  ```
- Evaluate:
  ```bash
  python -m training.launch evaluate --config config/base.yaml --checkpoint checkpoints/.../best_model.pt
  ```
- Profile/benchmark:
  ```bash
  python -m training.launch profile --config config/base.yaml --profile-steps 20 --benchmark
  ```

## Coordinator and Redis
- Set `REDIS_URL` for Redis-backed pool.
- File fallback at `configs/decoder_pool.json` is auto-mirrored.
- Mirror interval via `COORDINATOR_MIRROR_INTERVAL` (default 60s, min 5s). Logged at startup.

## Troubleshooting
- Dependency check:
  ```bash
  python scripts/check_dependencies.py
  ```
- CUDA available:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Decoder health:
  ```bash
  curl -v http://localhost:8001/health
  ```
More in [TROUBLESHOOT.md](TROUBLESHOOT.md).

## Resources
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API.md](API.md)
- [TRAINING.md](TRAINING.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SDK_INTEGRATION.md](SDK_INTEGRATION.md)
- [REDIS_INTEGRATION.md](REDIS_INTEGRATION.md)
- [DECODER_POOL.md](DECODER_POOL.md)
- [environment-variables.md](environment-variables.md)
- [TROUBLESHOOT.md](TROUBLESHOOT.md)
- [Contributing](../CONTRIBUTING.md)