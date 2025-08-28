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
- Docker (optional, for local stack)
- CUDA GPU (optional, for training/decoder)

### Quick Start
```bash
# Clone
git clone https://github.com/code-with-zeeshan/universal-translation-system
cd universal-translation-system

# Create virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install
pip install -r requirements.txt

# Validate system
python main.py --mode setup --validate-only --verbose

# Run tests
pytest tests/
```

### Local stack (Docker Compose)
```bash
docker compose up -d --build encoder decoder redis coordinator prometheus grafana
# Encoder:     http://localhost:8000
# Decoder:     http://localhost:8001
# Coordinator: http://localhost:8002
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000
```

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
   python -m training.launch train --config config/training_generic_gpu.yaml
   ```
   - Deprecated: `training/train_universal_system.py` (use the launcher instead)

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
- [TROUBLESHOOT.md](TROUBLESHOOT.md)
- [Contributing](../CONTRIBUTING.md)