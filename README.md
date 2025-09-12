# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include an innovative edge-cloud architecture, customizable language support, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

> **New**: All configuration is now available through environment variables. See [Environment Variables](docs/environment-variables.md) for details.

## 🌟 Key Innovation

Rather than bundling a huge model per language, the system splits the workflow for maximum efficiency and scalability:
- **Edge Universal Encoder (~35MB)**: Converts text to language-agnostic embeddings with RoPE and SwiGLU for speed/quality on-device.
- **On‑demand Vocabulary Packs (2–4MB)**: Download only what you need (groups like latin, cjk, cyrillic, arabic, devanagari, thai), compressed with LZ4/Zstandard and loaded dynamically.
- **Cloud Decoder (6-layer transformer)**: Cross‑attention decoder served via Litserve for high throughput; supports dynamic adapter loading per language/domain.
- **Smart Coordinator**: Routes to least‑loaded decoders, performs health checks, supports elastic scaling, exposes Prometheus metrics and Grafana dashboards.
- **Multi‑SDK + WebAssembly**: Native Android/iOS/Flutter/React Native, and Web with WASM; automatic fallback to cloud when a device lacks resources.
- **Result**: ~40MB app footprint with ~90% of full model quality, works on 2GB RAM devices, and scales in the cloud for quality and throughput.

## 📋 Features

- ✅ 20 language support with dynamic vocabulary loading
- ✅ Native SDKs for Android, iOS, Flutter, React Native, and Web
- ✅ Edge encoding, cloud decoding architecture
- ✅ ~85M parameters total (vs 600M+ for traditional models)
- ✅ Designed for low-end devices (2GB RAM)
- ✅ Full-system monitoring with Prometheus/Grafana
- ✅ Environment variable configuration for all components
- ✅ Docker and Kubernetes deployment support
- ✅ Redis integration for distributed decoder pool management
- ✅ Advanced memory management with automatic cleanup
- ✅ Comprehensive profiling system for performance optimization
- ✅ Configurable HTTPS enforcement with security headers
- ✅ Bottleneck detection and performance analysis

## 🎯 Usage Modes

You can use `universal-decoder-node` in three ways:

- **Personal Use:**  
  Run the decoder on your own device or cloud for private translation needs and testing. No registration is required.

- **Contributing Compute Power:**  
  If you want to support the project and make your node available to the global system, register your node with the coordinator so it can be added to the public decoder pool.

- **Hybrid Deployment:**  
  Run your own encoder locally while using the shared decoder pool, or run your own complete system with encoder, decoder, and coordinator.

See [CONTRIBUTING.md](CONTRIBUTING.md) for registration instructions and [REDIS_INTEGRATION.md](docs/REDIS_INTEGRATION.md) for details on the distributed decoder pool.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system

# Set up environment variables (recommended)
cp .env.example .env
# Edit .env with your configuration

# Option 1: Run with Docker Compose (recommended)
docker-compose up -d

# Option 2: Manual Setup
# Install dependencies
pip install -r requirements.txt

# Run components individually
python cloud_decoder/optimized_decoder.py
python coordinator/advanced_coordinator.py

# Option 3: Train from scratch (for research)
python docs/train_from_scratch.py --config config/training_default.yaml
```

## 🧭 Unified Pipeline CLI

Use a single entrypoint to run data, vocabulary, training, and more.

- Location: `scripts/pipeline.py`
- Prereq: `pip install -r requirements.txt`

Examples:

```bash
# Data pipeline (all stages)
python scripts/pipeline.py data --config config/training_generic_gpu.yaml

# Data pipeline (specific stages)
# Valid stages: download_evaluation download_training sample_filter augment create_ready validate vocabulary
python scripts/pipeline.py data --config config/training_generic_gpu.yaml --stages download_training create_ready

# Vocabulary creation
python scripts/pipeline.py vocab --mode production --corpus-dir data/processed --output-dir vocabs

# Bootstrap pretrained encoder/decoder
python scripts/pipeline.py bootstrap --encoder-model xlm-roberta-base --decoder-model facebook/mbart-large-50

# Train (delegates to training.launch)
python scripts/pipeline.py train --config config/training_generic_gpu.yaml --distributed

# Evaluate, Profile, Compare
python scripts/pipeline.py evaluate --config config/training_generic_gpu.yaml --checkpoint models/.../best_model.pt
python scripts/pipeline.py profile --config config/training_generic_gpu.yaml --profile-steps 25 --benchmark
python scripts/pipeline.py compare --experiments runs/exp1 runs/exp2

# Convert models
python scripts/pipeline.py convert --task pytorch-to-onnx --model-path models/encoder/universal_encoder_initial.pt --output-path models/encoder/universal_encoder.onnx

# Full pipeline: data -> vocab -> bootstrap -> train -> convert
python scripts/pipeline.py all --config config/training_generic_gpu.yaml
```

Notes:
- Data stages map to the orchestrator in `data.unified_data_pipeline.UnifiedDataPipeline`.
- Vocabulary uses `vocabulary.unified_vocabulary_creator` with modes/groups.
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
# Data pipeline (all stages)
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" data --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml"

# Data pipeline (specific stages)
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" data --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml" --stages download_training create_ready

# Vocabulary creation
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" vocab --mode production --corpus-dir "c:\Users\DELL\universal-translation-system\data\processed" --output-dir "c:\Users\DELL\universal-translation-system\vocabs"

# Bootstrap pretrained encoder/decoder
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" bootstrap --encoder-model xlm-roberta-base --decoder-model facebook/mbart-large-50

# Train (delegates to training.launch)
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" train --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml" --distributed

# Evaluate, Profile, Compare
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" evaluate --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml" --checkpoint "c:\Users\DELL\universal-translation-system\models\...\best_model.pt"
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" profile --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml" --profile-steps 25 --benchmark
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" compare --experiments "c:\Users\DELL\universal-translation-system\runs\exp1" "c:\Users\DELL\universal-translation-system\runs\exp2"

# Convert models
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" convert --task pytorch-to-onnx --model-path "c:\Users\DELL\universal-translation-system\models\encoder\universal_encoder_initial.pt" --output-path "c:\Users\DELL\universal-translation-system\models\encoder\universal_encoder.onnx"

# Full pipeline: data -> vocab -> bootstrap -> train -> convert
python "c:\Users\DELL\universal-translation-system\scripts\pipeline.py" all --config "c:\Users\DELL\universal-translation-system\config\training_generic_gpu.yaml"
```

## 📱 SDK Integration

See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for full details and code examples for all platforms.

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

## 🏗️ Architecture

- **Encoder**: Runs on device, converts text to language-agnostic embeddings
- **Decoder**: Runs on server (Litserve), converts embeddings to target language
- **Coordinator**: Manages decoder pool, handles load balancing and health monitoring
- **Vocabulary Packs**: Downloadable language-specific token mappings
- **Model Weights**: Shared between all languages, trained on a diverse corpus

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 📚 Documentation

### Quick Links
- SDKs: [SDK Integration Guide](docs/SDK_INTEGRATION.md) | Publishing: [SDK_PUBLISHING.md](docs/SDK_PUBLISHING.md)
- APIs: [API Documentation](docs/API.md)
- Deployment: [Deployment Guide](docs/DEPLOYMENT.md)
- Decoder Pool: [Decoder Pool Management](docs/DECODER_POOL.md)
- Environment: [Environment Variables](docs/environment-variables.md)

### Full Docs
- [Vision & Architecture](docs/VISION.md)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Environment Variables](docs/environment-variables.md)
- [Training Guide](docs/TRAINING.md)
- [Adding New Languages](docs/Adding_New_languages.md)
- [Future Roadmap](docs/future_plan.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [SDK Integration Guide](docs/SDK_INTEGRATION.md)
- [CI: Build & Upload to HF](docs/CI_BUILD_UPLOAD.md)
- [Monitoring Guide](monitoring/README.md)
- [Vocabulary Guide](docs/Vocabulary_Guide.md)
- [Decoder Pool Management](docs/DECODER_POOL.md)
- [API Documentation](docs/API.md)
- [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)
- [Troubleshooting Guide](docs/TROUBLESHOOT.md)
- [Security Best Practices](docs/SECURITY_BEST_PRACTICES.md) — includes secret bootstrap, *_FILE usage, and rotation workflow
- [System Improvements](docs/IMPROVEMENTS.md)
- [Acknowledgments](docs/ACKNOWLEDGMENTS.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [License](LICENSE)

## 🔐 Secrets & Redis (Coordinator)

### Redis (for rate limiting and token revocation)
- Set environment and connection (example):
  - `REDIS_URL=redis://localhost:6379/0`
  - Or `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- Coordinator auto-initializes Redis if available.
- Revocation: jti values are stored in Redis set `revoked_jti`. Endpoint `POST /api/revoke` adds a jti (admin session required).
- If Redis is unavailable, revocation falls back to allow (recommended to run Redis in production).

## 🔐 Secrets Management (Docker Compose & Kubernetes)

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

Example (already wired in docker-compose.yml):
- Coordinator uses:
  - `COORDINATOR_SECRET_FILE`, `COORDINATOR_JWT_SECRET_FILE`, `COORDINATOR_TOKEN_FILE`, `INTERNAL_SERVICE_TOKEN_FILE`
- Decoder uses:
  - `DECODER_JWT_SECRET_FILE`

### Kubernetes
- Create/update `kubernetes/secrets.yaml` with base64-encoded values.
- Apply secrets and deployments:
```bash
kubectl apply -f kubernetes/secrets.yaml
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

## 📊 Monitoring
- All services expose Prometheus metrics at `/metrics`
- Visualize with Grafana dashboards (included in `monitoring/grafana/dashboards`)
- Set up alerts for latency, errors, and resource usage
- See [monitoring/README.md](monitoring/README.md) for details

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and more details on how to contribute.

## ⚠️ Current Status
This is a research project in active development. Core components are implemented but not production-tested.

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Encoder | ✅ Production-Ready | Core functionality complete and tested |
| Decoder | ✅ Production-Ready | Core functionality complete and tested |
| Vocabulary System | ✅ Production-Ready | Supports all planned languages |
| Coordinator | ✅ Production-Ready | Load balancing and health monitoring implemented |
| Android SDK | ✅ Production-Ready | Native implementation with JNI bindings |
| iOS SDK | ✅ Production-Ready | Swift implementation with C++ interoperability |
| Flutter SDK | ✅ Production-Ready | FFI bindings to native encoder |
| React Native SDK | ✅ Production-Ready | Core functionality implemented with config support |
| Web SDK | ✅ Production-Ready | Core functionality implemented with environment variable support |
| Monitoring | ✅ Production-Ready | Prometheus metrics and health checks implemented |
| Docker Support | ✅ Production-Ready | Docker Compose and Kubernetes configurations available |
| Environment Config | ✅ Production-Ready | All components configurable via environment variables |

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We are grateful to the amazing open-source community and the researchers who have made this project possible. See [ACKNOWLEDGMENTS.md](docs/ACKNOWLEDGMENTS.md) for detailed acknowledgments.

## 📜 Changelog
See [CHANGELOG.md](CHANGELOG.md) for the latest updates.