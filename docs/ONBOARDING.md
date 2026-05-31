# Onboarding Guide (Developers and Operators)

Welcome to the Universal Translation System! This guide helps both contributors and operators get set up and run the system quickly and correctly.

## Table of Contents
1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Key Components](#key-components)
5. [Common Workflows](#common-workflows)
6. [Coordinator and Redis](#coordinator-and-redis)
7. [Troubleshooting](#troubleshooting)
8. [Resources](#resources)

## System Overview
- **Edge (Client)**: Lightweight universal encoder (35MB base + 2-4MB vocabulary packs). Optional language adapters (~2MB each) for domain/language tuning.
- **Cloud**: Optimized Universal Decoder (6-layer transformer, FastAPI/uvicorn) returns translations; supports hot-loaded adapters and runtime vocab access.
- **Coordinator**: Routes requests to least-loaded healthy decoders; manages pool via Redis with periodic mirroring to disk; exposes metrics and admin UI.

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

### Quick Start Using Install Script
```bash
# Development setup with all extras
bash scripts/install.sh --dev

# Or for serving only
bash scripts/install.sh --serve
```

### CPU-only Demo (no Docker)
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using role-based install
bash scripts/install.sh --dev

# Validate setup (creates missing dirs and checks dependencies)
python main.py --mode setup --validate-only --verbose

# Run a quick translation (non-interactive)
python main.py --mode translate --text "Hello world" --source-lang en --target-lang es

# Optional: smoke test utilities
python scripts/smoke_cpu.py
```

### Full Local Stack (Docker Compose)

#### 1) Configure environment (CRITICAL)
```bash
cp .env.example .env
# Edit .env and set these mandatory values:
# - DECODER_JWT_SECRET or DECODER_JWT_SECRET_FILE
# - COORDINATOR_SECRET or COORDINATOR_SECRET_FILE
# - COORDINATOR_JWT_SECRET or COORDINATOR_JWT_SECRET_FILE
# - COORDINATOR_TOKEN or COORDINATOR_TOKEN_FILE
# - INTERNAL_SERVICE_TOKEN or INTERNAL_SERVICE_TOKEN_FILE
# - (Optional) JWT_PUBLIC_KEY_PATH and JWT_PRIVATE_KEY_FILE for RS256
# - HF_HUB_REPO_ID (and HF_TOKEN if private)
```
- Secrets can be provided via `*_FILE` envs in Docker/Kubernetes.
- If you change `.env`, restart compose or run with `--env-file .env`.

#### Quick .env template (local development)
```env
DECODER_JWT_SECRET=replace-with-strong-random
COORDINATOR_SECRET=replace-with-strong-random
COORDINATOR_JWT_SECRET=replace-with-strong-random
COORDINATOR_TOKEN=replace-with-strong-random
INTERNAL_SERVICE_TOKEN=replace-with-strong-random
REDIS_PASSWORD=replace-with-strong-random
HF_HUB_REPO_ID=code-with-zeeshan/universal-translation-system
HF_TOKEN=your-hf-token
JWT_ISS=local
JWT_AUD=universal-translation
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
JWT_PUBLIC_KEY_PATH=./config/jwt_public.pem
JWT_PRIVATE_KEY_FILE=./config/jwt_private.pem
```

#### Local test tokens (non-production)
```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

#### RS256 key pair (optional, local testing)
```bash
openssl genrsa -out config/jwt_private.pem 2048
openssl rsa -in config/jwt_private.pem -pubout -out config/jwt_public.pem
```

#### 2) Prefetch artifacts (recommended before starting services)
```bash
python tools/prefetch_artifacts.py \
  --repo_id code-with-zeeshan/universal-translation-system \
  --packs latin cjk \
  --adapters es fr \
  --models production/encoder.onnx
```

#### 3) Start services
```bash
docker compose --env-file .env up -d --build encoder decoder redis coordinator prometheus grafana
# Encoder:     http://localhost:8000
# Decoder:     http://localhost:8001
# Coordinator: http://localhost:8002
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000
```

#### 4) Validate services
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### Model Artifacts & Paths
- **Models (local)**: `./models`
  - `models/production/decoder.pt` (default expected by decoder)
  - `models/production/encoder.pt` (default expected by encoder)
  - `models/model_registry.json` (optional)
- **Vocabulary (local)**: `./vocabs` (mounted to `/app/vocabs` in containers)
  - Contains vocabulary packs and optional `manifest.json`
- **Checkpoints**: `./checkpoints/<experiment_name>/`
- All paths overridable via `UTS_*` env vars in `utils/constants.py`

See also: `docs/DEPLOYMENT.md`, `docs/Vocabulary_Guide.md`, `docs/API.md`.

#### GPU notes
- Ensure NVIDIA drivers + NVIDIA Container Toolkit are installed for decoder GPU support.
```bash
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Training for Production
```bash
# Recommended: unified launcher (single-GPU/CPU)
python -m training.launch train --config config/base.yaml

# Force distributed on multiple GPUs
python -m training.launch train --config config/base.yaml --distributed

# Evaluate a checkpoint
python -m training.launch evaluate --config config/base.yaml --checkpoint checkpoints/.../best_model.pt

# Export for deployment
python scripts/pipeline.py convert --task pytorch-to-onnx --model-path models/encoder/universal_encoder_initial.pt --output-path models/encoder/universal_encoder.onnx
```
- After exporting, copy artifacts to `models/production/` and `vocabs/`, or publish to HF Hub.

#### Optional: Warm-start (Bootstrap) initial weights
```bash
python -c "from training.bootstrap_from_pretrained import PretrainedModelBootstrapper as B; b=B(); b.create_encoder_from_pretrained('xlm-roberta-base', 'models/encoder/universal_encoder_initial.pt', 1024)"
python -c "from training.bootstrap_from_pretrained import PretrainedModelBootstrapper as B; b=B(); b.create_decoder_from_mbart('facebook/mbart-large-50', 'models/decoder/universal_decoder_initial.pt')"
```
- If these files exist, the launcher auto-loads them:
  - `models/encoder/universal_encoder_initial.pt`
  - `models/decoder/universal_decoder_initial.pt`

### Production Deployment Checklist
- **Secrets**: set strong values for `DECODER_JWT_SECRET`, `COORDINATOR_JWT_SECRET`, `COORDINATOR_TOKEN`, `INTERNAL_SERVICE_TOKEN`.
- **Artifacts**: publish models/vocabulary/adapters to HF Hub; ensure access tokens configured.
- **GPU**: GPU nodes provisioned; NVIDIA runtime installed.
- **Persistence**: volumes/PVCs for `models/`, `vocabs/`, Redis data.
- **Networking**: TLS termination, auth on public endpoints, proper CORS.
- **Monitoring**: Prometheus + Grafana dashboards and alerts.
- See: `docs/DEPLOYMENT.md`, `docs/DECODER_POOL.md`, `charts/uts/`.

### Common Pitfalls
- Docker Compose env: do not rely on shell expansions in compose; use `.env` variables.
- GPU on Windows: prefer WSL2 + NVIDIA toolkit.
- HF Hub: private repos require `HF_TOKEN` available to services.
- Missing artifacts: prefetch models/vocabs before `docker compose up`.

## Development Environment Setup

### Python
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements/base.txt -r requirements/train.txt -r requirements/serve.txt
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
- `encoder/` -- Python encoder (training-time)
- `encoder_core/` -- C++ encoder core (deployment)
- `vocabulary/` -- Vocabulary management (`vocabulary_creator.py`, `vocab_production.py`, `vocab_research.py`, `vocab_validation.py`, `vocab_config.py`)
- `cloud_decoder/` -- Server-side decoder (FastAPI/uvicorn)
- `coordinator/` -- Advanced routing, health, metrics, Redis-backed pool
- `monitoring/` -- Prometheus/Grafana configs and collectors
- `training/` -- Training system (`trainer.py`, `launch.py`, `hardware_profile.py`, `memory_trainer.py`, `encoder_quantizer.py`, etc.)
- `data/` -- Data pipeline (`pipeline_orchestrator.py`, `pipeline_state.py`, `custom_samplers.py`)
- `integration/` -- System wiring (`system.py`, `system_config.py`, `system_health.py`, `translation_api.py`)
- `evaluation/` -- Evaluation (`evaluator.py`, `metrics.py`)
- **SDKs**: `sdk/android/`, `sdk/ios/`, `sdk/flutter/`, `sdk/react-native/`, `sdk/web/`
- `utils/` -- Shared utilities (`constants.py` with UTS_* path overrides, `redis_manager.py`, `thread_safety.py`, `logging_config.py`)
- `config/` -- YAML configs + `schemas.py` (canonical config hierarchy)

## Common Workflows

### 1) Add a new language
1. Prepare data
2. Update/create vocabulary:
   ```bash
   python -c "from vocabulary.vocabulary_creator import UnifiedVocabularyCreator, CreationMode; c=UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabs'); c.create_pack(pack_name='latin', languages=['en','es','fr','de','it'], mode=CreationMode.PRODUCTION)"
   ```
3. Train with the unified launcher:
   ```bash
   python -m training.launch train --config config/base.yaml
   ```

### 2) Modify Encoder
1. Change code in `encoder/` or C++ in `encoder_core/`
2. Tests:
   ```bash
   pytest tests/test_encoder.py
   ```
3. Build C++ core:
   ```bash
   bash scripts/build_encoder_core.sh
   ```

### 3) Modify Decoder
1. Change code in `cloud_decoder/`
2. Tests:
   ```bash
   pytest tests/test_decoder.py
   ```

### 4) Training/Evaluation/Profile
```bash
python -m training.launch train --config config/base.yaml
python -m training.launch evaluate --config config/base.yaml --checkpoint checkpoints/.../best_model.pt
python -m training.launch profile --config config/base.yaml --profile-steps 20 --benchmark
```

## Coordinator and Redis
- Set `REDIS_URL` for Redis-backed pool.
- File fallback at `configs/decoder_pool.json` is auto-mirrored (default `COORDINATOR_MIRROR_INTERVAL`=60s).
- See also `scripts/setup_redis.sh` for Redis installation with Docker fallback.

## Troubleshooting
```bash
# Dependency check
python scripts/check_dependencies.py

# CUDA check
python -c "import torch; print(torch.cuda.is_available())"

# Decoder health
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
