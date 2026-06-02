# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include an innovative edge-cloud architecture, customizable language support, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

> **New**: All configuration is now available through environment variables. See [Environment Variables](docs/environment-variables.md) for details.

## Key Innovation

Rather than bundling a huge model per language, the system splits the workflow for maximum efficiency and scalability:
- **Edge Universal Encoder (~25MB)**: Compact 768-dim, 6-layer, 8-head encoder initialized from XLM-RoBERTa-base for perfect zero-loss weight transfer. RoPE + SwiGLU for speed/quality on-device.
- **On-demand Vocabulary Packs (2–4MB)**: 32K BPE vocabulary via SentencePiece, grouped by script (latin, cjk, cyrillic, arabic, devanagari, thai), compressed with LZ4/Zstandard.
- **Cloud Decoder (6-layer, 512-dim, 8-head)**: Cross-attention decoder initialized from mBART-large-50, served via FastAPI/uvicorn. 512-dim provides robust capacity while keeping ~20MB footprint.
- **PEFT LoRA Training**: Backbone (XLM-RoBERTa + mBART) frozen during training; only ~5-10M LoRA adapter parameters trained per run. This reduces GPU training to **~4-6 hours on an L4** (vs 5-7 days for full fine-tune), with minimal quality loss.
- **Smart Coordinator**: Routes to least-loaded decoders, performs health checks, supports elastic scaling, exposes Prometheus metrics and Grafana dashboards.
- **Multi-SDK + WebAssembly**: Native Android/iOS/Flutter/React Native, and Web with WASM; automatic fallback to cloud when a device lacks resources.
- **Result**: ~40MB app footprint with ~90% of full model quality, works on 2GB RAM devices, and scales in the cloud for quality and throughput.

## Features

- 20 language support with dynamic vocabulary loading (32K BPE, SentencePiece)
- Native SDKs for Android, iOS, Flutter, React Native, and Web (under `sdk/`)
- Edge encoding, cloud decoding architecture
- **Encoder**: XLM-RoBERTa-base (768 hidden, perfect weight transfer) + **Decoder**: mBART-large-50 (512 decoder dim)
- **PEFT LoRA training**: Freeze backbone, train only adapters. ~5-10M trainable params vs 600M+ full fine-tune
- Progressive training in 4 tiers (3/2/2/1 epochs) for curriculum learning
- Designed for low-end devices (2GB RAM), ~40MB total footprint
- Full-system monitoring with Prometheus/Grafana
- Environment variable configuration for all components
- Docker and Kubernetes deployment support
- Redis integration for distributed decoder pool management
- Advanced memory management with automatic cleanup
- Comprehensive profiling system for performance optimization
- Configurable HTTPS enforcement with security headers
- Bottleneck detection and performance analysis

## Usage Modes

You can use `universal-decoder-node` in three ways:

- **Personal Use:**  
  Run the decoder on your own device or cloud for private translation needs and testing. No registration is required.

- **Contributing Compute Power:**  
  If you want to support the project and make your node available to the global system, register your node with the coordinator so it can be added to the public decoder pool.

- **Hybrid Deployment:**  
  Run your own encoder locally while using the shared decoder pool, or run your own complete system with encoder, decoder, and coordinator.

See [CONTRIBUTING.md](CONTRIBUTING.md) for registration instructions and [REDIS_INTEGRATION.md](docs/REDIS_INTEGRATION.md) for details on the distributed decoder pool.

## Quick Start

```bash
# Clone repository
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system

# Set up environment variables (recommended)
cp .env.example .env
# Edit .env with your configuration

# Option 1: Run with Docker Compose (recommended)
docker compose --env-file .env up -d

# Option 2: Manual Setup
# Install dependencies (modular)
# Base runtime
pip install -r requirements/base.txt
# Add training + serving
pip install -r requirements/train.txt -r requirements/serve.txt
# Optional service-specific extras
pip install -r requirements/decoder.txt -r requirements/coordinator.txt

# Option 3: Role-based install script
bash scripts/install.sh --serve   # --train, --coordinator, --dev, --encoder-core, --all

# Run components individually
python cloud_decoder/optimized_decoder.py
python coordinator/advanced_coordinator.py

# Training with LoRA (fast, recommended)

```bash
# 1. Bootstrap encoder/decoder from pretrained weights
python -m training.bootstrap_from_pretrained

# 2. Run progressive training (4 tiers, LoRA adapters)
python -m training.progressive_training

# 3. Upload to Hugging Face Hub
python scripts/upload_artifacts.py --repo_id your-username/uts-models
```

LoRA is enabled by default (`use_lora: true` in `config/base.yaml`). The backbone (XLM-RoBERTa + mBART) stays frozen; only ~5M adapter params are trained. This completes in **~4-6 hours on an L4 GPU** at ~$0.48/hr.

### Training Cost Guide (Lightning AI Studio)

| GPU | $/hr | Total run | Quality |
|---|---|---|---|
| **L4 (24GB)** | $0.48 | **~$2-3** | Target quality ✓ |
| T4 (16GB) | $0.19 | ~$1 | Slightly slower |
| L40s (48GB) | $1.79 | ~$8 | Overkill for LoRA |
| A100 (40GB) | $1.55 | ~$7 | Overkill for LoRA |

### LoRA Rank Tuning

| LoRA r | Trainable params | VRAM | Quality | Time |
|---|---|---|---|---|
| r=4 | ~2.5M | ~1.5GB | Good | ~3h |
| **r=8** | **~5M** | **~2GB** | **Better** | **~4h** |
| r=16 | ~10M | ~2.5GB | Best | ~5h |

Set via `config/base.yaml`: `lora_r: 16`, `lora_alpha: 32` for best quality.

**CI: Enforce schema hash up-to-date**

Add a step to your CI pipeline to ensure version-config.json is regenerated when schemas change.

Example GitHub Actions step:

```yaml
- name: Verify schema hash is up-to-date
  shell: bash
  run: |
    python scripts/update_schema_hash.py
    git diff --exit-code version-config.json || {
      echo "version-config.json schemaHash drifted. Run 'python scripts/update_schema_hash.py' and commit the change.";
      exit 1;
    }
```

## Unified Pipeline CLI

Use a single entrypoint to run data, vocabulary, training, and more.

- Location: `scripts/pipeline.py`
- Prereq: `pip install -r requirements/base.txt -r requirements/train.txt -r requirements/serve.txt` (add service extras as needed)

Examples (run from repo root):

```bash
# Data pipeline (all stages)
python ./scripts/pipeline.py data --config ./config/base.yaml

# Data pipeline (specific stages)
python ./scripts/pipeline.py data --config ./config/base.yaml --stages download_training create_ready

# Vocabulary creation
python ./scripts/pipeline.py vocab --mode production --corpus-dir ./data/processed --output-dir vocabulary/vocab

# Bootstrap pretrained encoder/decoder
python ./scripts/pipeline.py bootstrap --encoder-model xlm-roberta-base --decoder-model facebook/mbart-large-50

# Train (delegates to training.launch)
python ./scripts/pipeline.py train --config ./config/base.yaml --distributed

# Evaluate, Profile, Compare
python ./scripts/pipeline.py evaluate --config ./config/base.yaml --checkpoint ./models/.../best_model.pt
python ./scripts/pipeline.py profile --config ./config/base.yaml --profile-steps 25 --benchmark
python ./scripts/pipeline.py compare --experiments ./runs/exp1 ./runs/exp2

# Convert models
python ./scripts/pipeline.py convert --task pytorch-to-onnx --model-path ./models/encoder/universal_encoder_initial.pt --output-path ./models/encoder/universal_encoder.onnx

# Full pipeline: data -> vocab -> bootstrap -> train -> convert
python ./scripts/pipeline.py all --config ./config/base.yaml
```

Notes:
- Data stages map to the orchestrator in `data.pipeline_orchestrator.UnifiedDataPipeline`.
- Vocabulary uses `vocabulary.vocabulary_creator` with modes/groups.
- Training/eval/profile/compare proxy to `training.launch`.
- Conversion wraps `training.convert_models.ModelConverter` tasks.

### Quick help

```bash
# Top-level help
python scripts/pipeline.py --help

# Subcommand help
python scripts/pipeline.py data --help
python scripts/pipeline.py vocab --help
python scripts/pipeline.py train --help
```

### Windows PowerShell examples

```powershell
# Run from repository root (cloned anywhere). Use absolute path via $PWD for reliability.
python "$PWD\scripts\pipeline.py" data --config "$PWD\config\base.yaml"
python "$PWD\scripts\pipeline.py" vocab --mode production --corpus-dir "$PWD\data\processed" --output-dir "$PWD\vocabs"
python "$PWD\scripts\pipeline.py" bootstrap --encoder-model xlm-roberta-base --decoder-model facebook/mbart-large-50
python "$PWD\scripts\pipeline.py" train --config "$PWD\config\base.yaml" --distributed
python "$PWD\scripts\pipeline.py" evaluate --config "$PWD\config\base.yaml" --checkpoint "$PWD\models\...\best_model.pt"
python "$PWD\scripts\pipeline.py" profile --config "$PWD\config\base.yaml" --profile-steps 25 --benchmark
python "$PWD\scripts\pipeline.py" compare --experiments "$PWD\runs\exp1" "$PWD\runs\exp2"
python "$PWD\scripts\pipeline.py" convert --task pytorch-to-onnx --model-path "$PWD\models\encoder\universal_encoder_initial.pt" --output-path "$PWD\models\encoder\universal_encoder.onnx"
python "$PWD\scripts\pipeline.py" all --config "$PWD\config\base.yaml"
```

## Model Artifacts & Paths
- **Models (local):** `./models`
  - Defaults: `./models/production/encoder.pt`, `./models/production/decoder.pt`
  - Artifact registry: `./models/model_registry.json`
- **Vocabulary (local):** `vocabulary/vocab/` (mounted to `/app/vocabs` in containers)
  - Packs and optional `manifest.json`
- **Checkpoints:** `./checkpoints/<experiment_name>/`

All paths are overridable via `UTS_*` environment variables defined in `utils/constants.py`.

### Hugging Face Hub Upload

The system auto-uploads models and vocabulary packs to Hugging Face Hub when configured:

```bash
# 1. Login
huggingface-cli login
# or set token
export HF_TOKEN=hf_your_token_here

# 2. Upload all artifacts (models, vocabs, adapters)
python scripts/upload_artifacts.py --repo_id your-username/uts-models

# 3. Or let ModelVersion auto-upload during register_model()
# Set in env:
export HF_HUB_REPO_ID=your-username/uts-models
```

The `ModelVersion` class in `utils/model_versioning.py` handles:
- Model registration with SHA-256 hashes and HMAC signing
- Automatic HF Hub upload on `register_model()`
- Version pinning for serving, rollback, and canary→production promotion
- Vocabulary packs are uploaded separately via `upload_artifacts.py` under `vocabulary/vocab/`

See also: [docs/ONBOARDING.md](docs/ONBOARDING.md) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## SDK Integration

See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for full details and code examples for all platforms.

All SDKs live under `sdk/`:
- `sdk/android/UniversalTranslationSDK/`
- `sdk/ios/UniversalTranslationSDK/`
- `sdk/flutter/universal_translation_sdk/`
- `sdk/react-native/UniversalTranslationSDK/`
- `sdk/web/universal-translation-sdk/`

### Android
```java
val translator = TranslationClient(context)
val result = translator.translate("Hello", "en", "es")
```

### iOS
```swift
let translator = TranslationClient()
let result = try await translator.translate(text: "Hello", from: "en", to: "es")
```

### Web
```javascript
const translator = new TranslationClient();
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "es"
});
```

### React Native
```javascript
import { TranslationClient } from 'universal-translation-sdk';

const translator = new TranslationClient();
const result = await translator.translate({
  text: "Hello",
  sourceLang: "en",
  targetLang: "es"
});
```

## Architecture (v2 - Compact LoRA-based)

- **Encoder**: 768 hidden, 6 layers, 8 heads, RoPE + SwiGLU. Initialized from XLM-RoBERTa-base (perfect 768→768 dimension match, zero-loss transfer). ~25MB.
- **Decoder**: 512 decoder dim, 6 layers, 8 heads. Initialized from mBART-large-50 with PCA adaptation (1024→512). ~20MB.
- **Training**: Backbone frozen via PEFT LoRA (r=4/8/16). Only LoRA adapters trained (~1-2% of params). Progressive 4-tier curriculum with reduced epochs (3/2/2/1).
- **Coordinator**: Manages decoder pool, handles load balancing and health monitoring
- **Vocabulary Packs**: 32K BPE vocabulary via SentencePiece, grouped by script (6 groups, 2-4MB per pack)
- **Model Weights**: Shared between all languages, trained on a diverse corpus

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Documentation

### Quick Links
- SDKs: [SDK Integration Guide](docs/SDK_INTEGRATION.md) | Publishing: [SDK_PUBLISHING.md](docs/SDK_PUBLISHING.md)
- APIs: [API Documentation](docs/API.md)
- Deployment: [Deployment Guide](docs/DEPLOYMENT.md)
- Environment: [Environment Variables](docs/environment-variables.md)
- Onboarding: [Onboarding Guide](docs/ONBOARDING.md)

### Full Docs
- [Vision & Architecture](docs/VISION.md)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Onboarding Guide](docs/ONBOARDING.md)
- [Environment Variables](docs/environment-variables.md)
- [Training Guide](docs/TRAINING.md)
- [Vocabulary Guide](docs/Vocabulary_Guide.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Decoder Pool Management](docs/DECODER_POOL.md)
- [CI: Build & Upload to HF](docs/CI_BUILD_UPLOAD.md)
- [Monitoring Guide](monitoring/README.md)
- [API Documentation](docs/API.md)
- [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)
- [Troubleshooting Guide](docs/TROUBLESHOOT.md)
- [Security Best Practices](docs/SECURITY_BEST_PRACTICES.md)
- [System Improvements](docs/IMPROVEMENTS.md)
- [Acknowledgments](docs/ACKNOWLEDGMENTS.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [License](LICENSE)

## Secrets & Redis (Coordinator)

### Redis (for rate limiting and token revocation)
- Set environment and connection (example):
  - `REDIS_URL=redis://localhost:6379/0`
  - Or `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- Coordinator auto-initializes Redis if available via `utils.redis_manager.RedisManager`.
- Revocation: jti values are stored in Redis set `revoked_jti`. Endpoint `POST /api/revoke` adds a jti (admin session required).
- If Redis is unavailable, revocation falls back to allow (recommended to run Redis in production).

### Secrets Bootstrap & Validation
- Centralized bootstrap loads `*_FILE` secrets into env and validates required keys at startup.
- Coordinator and Decoder now use the unified bootstrap and fail fast with actionable errors.
- See: docs/environment-variables.md (Secret Bootstrap & Validation) and tools/rotate_secrets.py for rotation.

### Docker Compose
- Place secret files under `../secrets/` relative to `docker/docker-compose.yml`:
  - `../secrets/decoder_jwt_secret.txt`
  - `../secrets/coordinator_secret.txt`
  - `../secrets/coordinator_jwt_secret.txt`
  - `../secrets/coordinator_token.txt`
  - `../secrets/internal_service_token.txt`
- Compose mounts them as Docker secrets. Services read via `*_FILE` envs.
- RS256 keys:
  - Put `../secrets/jwt_private_key.pem` and `../secrets/jwt_public_key.pem`
  - Set env for coordinator/decoder:
    - `JWT_PRIVATE_KEY_FILE=/run/secrets/jwt_private_key`
    - `JWT_PUBLIC_KEY_PATH=/run/secrets/jwt_public_key` (supports `||`-separated paths for rotation)

### Kubernetes
- Create/update `kubernetes/secrets.example.yaml` with base64-encoded values.
- Apply secrets and deployments:
```bash
kubectl apply -f kubernetes/secrets.example.yaml
kubectl apply -f kubernetes/coordinator-deployment.yaml
kubectl apply -f kubernetes/decoder-deployment.yaml
```
- Coordinator and Decoder mount a secret volume at `/var/run/secrets/uts` and use `*_FILE` envs.
- RS256 keys:
  - Add to `translation-system-secrets`:
    - `jwt-private-key` (PEM), `jwt-public-key` (PEM)
  - Mount at `/var/run/secrets/uts` and set envs:
    - `JWT_PRIVATE_KEY_FILE=/var/run/secrets/uts/jwt-private-key`
    - `JWT_PUBLIC_KEY_PATH=/var/run/secrets/uts/jwt-public-key`

### Base64 helper for Kubernetes
```bash
# Linux/macOS
printf 'my-secret' | base64
# Windows PowerShell
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('my-secret'))
```

Notes:
- Never commit plaintext secrets.
- Prefer separate keys per environment.
- For RS256, rotation is supported via multiple public keys using `JWT_PUBLIC_KEY` or `JWT_PUBLIC_KEY_PATH` with `||`-separated entries.

## Monitoring
- All services expose Prometheus metrics at `/metrics`
- Visualize with Grafana dashboards (included in `monitoring/grafana/dashboards`)
- Set up alerts for latency, errors, and resource usage
- See [monitoring/README.md](monitoring/README.md) for details

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and more details on how to contribute.

## Current Status
This is a research project in active development. Core components are implemented but not production-tested.

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | ✅ Stable | Download, sample, augment, create_ready, quality filter, validate |
| Vocabulary System | ✅ Stable | 6 language packs (latin, cjk, arabic, devanagari, cyrillic, thai) |
| Vocabulary Evolution | ✅ Implemented | Unknown-token promotion with model embedding retraining |
| Coordinator | ✅ Stable | Load balancing, Redis pool, health monitoring |
| Intelligent Training | ✅ Implementated | Hardware-aware FSDP/DDP/single, LoRA, progressive tiers |
| Encoder | Implemented | Pure Python, needs trained weights |
| Decoder | Implemented | Pure Python, needs trained weights |
| Android SDK | Scaffolded | Needs encoder binary |
| iOS SDK | Scaffolded | Needs encoder binary |
| Flutter SDK | Scaffolded | Needs encoder binary |
| React Native SDK | Scaffolded | Needs encoder binary |
| Web SDK | Scaffolded | WASM encoder is stub |
| Docker Support | ✅ Stable | Docker Compose, Kubernetes, and Helm chart |
| Monitoring | ✅ Stable | Prometheus/Grafana dashboards |
| C++/FFI Encoder Core | ⬜ Planned | For edge/on-device deployment |
| Voice/TTS Translation | ⬜ Planned | For future release |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We are grateful to the amazing open-source community and the researchers who have made this project possible. See [ACKNOWLEDGMENTS.md](docs/ACKNOWLEDGMENTS.md) for detailed acknowledgments.

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest updates.
