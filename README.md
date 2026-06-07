# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform designed to support multiple languages across diverse applications. This system enables seamless text translation, making it easy to localize content for global audiences. Features include an innovative edge-cloud architecture, customizable language support, and extensible modules for adding new languages or translation engines. Ideal for developers and organizations looking to streamline multilingual communication and content delivery.

> **New**: Unified CLI — run `./uts` to discover every tool by workflow.

## Key Innovation

Rather than bundling a huge model per language, the system splits the workflow for maximum efficiency and scalability:
- **Edge Universal Encoder (42.7M params)**: 512-dim, 6-layer, 8-head encoder with RoPE + SwiGLU for on-device inference.
- **On-demand Vocabulary Packs (2–4MB)**: 32K BPE vocabulary via SentencePiece, grouped by script (latin, cjk, cyrillic, arabic, devanagari, thai).
- **Cloud Decoder (108.1M params)**: 768-dim, 8-layer, 12-head cross-attention decoder, served via FastAPI.
- **Dual-Phase Training**: Phase 1 trains all 150.8M params for strong multilingual representations. Phase 2 freezes the backbone and trains only LoRA adapters (~7M params) when adding new languages — no full retraining needed.
- **Smart Coordinator**: Routes to least-loaded decoders, performs health checks, supports elastic scaling.
- **Multi-SDK**: Native Android/iOS/Flutter/React Native/Web SDKs under `sdk/`.

## Features

- 20 language support with dynamic vocabulary loading (32K BPE, SentencePiece)
- Native SDKs for Android, iOS, Flutter, React Native, and Web (under `sdk/`)
- Edge encoding, cloud decoding architecture
- **Dual-phase training**: Full backbone → LoRA adapters for new languages
- Full-system monitoring with Prometheus/Grafana
- Docker and Kubernetes deployment support
- Redis integration for distributed decoder pool management
- Comprehensive profiling system for performance optimization

## Quick Start

```bash
# Clone repository
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system

# One command to discover everything:
./uts

# Quick install + run on Lightning AI:
export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"
pip install -e ".[train]"
./uts data --pipeline            # Download & process data (~25 min)
./uts vocab --build              # Build vocabulary packs (~10 min)
./uts train --full               # Train full model (~6h on A100)
./uts eval --model               # Evaluate BLEU per language pair
```

See [SETUP_COMMANDS.md](SETUP_COMMANDS.md) for full GPU-tier-specific setup.

## Training Duration

Training time varies by GPU, batch size, and number of epochs.

| GPU | Full model (10 epochs) | Notes |
|---|---|---|
| A100 40GB / H100 | Fastest | Sweet spot for full training |
| L40s 48GB | Slightly faster | Overkill but quick |
| L4 24GB | Slower | Viable with smaller batch |
| T4 16GB | Slowest | Needs gradient checkpointing, long runtime |

**Rule of thumb:** Full model trains 3-5× faster than LoRA-on-random-init (which produces poor quality — always prefer full training).

See [SETUP_COMMANDS.md](SETUP_COMMANDS.md) for exact commands per GPU tier.

## Unified CLI: `./uts`

All tools are organized into 8 workflow groups. Run `./uts <group> --help` for details:

| Group | Purpose |
|---|---|
| `./uts setup` | Environment check, config wizard, validate config |
| `./uts data` | Download, sample, augment, validate pipeline |
| `./uts vocab` | Build or evolve vocabulary packs |
| `./uts train` | Full model training / progressive / LoRA |
| `./uts eval` | Evaluate model, benchmark, download test data |
| `./uts serve` | Start decoder server, coordinator, Redis |
| `./uts tools` | Config validation, GPU check, secrets, upload, prefetch |
| `./uts docs` | Open documentation by topic |

## Language Expansion Strategy

The system is designed for zero-full-retraining language expansion:

1. **Phase 1 (current):** Train full backbone on 20 languages (`use_lora: false` in config). Builds strong multilingual representations.
2. **Phase 2 (future):** Freeze backbone, set `use_lora: true`, train LoRA adapters + target language adapter for languages 21+. Only ~7M trainable params vs 150.8M.

```bash
# Phase 1: Full model training
./uts train --full                    # Default 5 epochs (~3h)
./uts train --full --num-epochs 10    # Full convergence (~6h)

# Phase 2: LoRA adapter for new language
# (edit config: add language, set use_lora: true)
./uts train --full --experiment-name "new-lang-adapter"
```

## Architecture (Current)

| Component | Params | Dim | Layers | Heads | Role |
|---|---|---|---|---|---|
| Edge Encoder | 42.7M | 512 | 6 | 8 | On-device inference |
| Cloud Decoder | 108.1M | 768 | 8 | 12 | Cross-attention decoding |
| Vocab Packs | 6 × 32K tokens | — | — | — | Per-script (latin, cjk, etc.) |

- **Encoder**: RoPE + SwiGLU, Pre-LN normalization. Dynamically resizes embedding to match loaded vocab pack.
- **Decoder**: Cross-attention to encoder hidden states. Per-language adapter bottlenecks (768→96→768) after each layer.
- **Training**: Full model, teacher forcing, label smoothing, bfloat16 mixed precision.

## Languages (20)

en, es, fr, de, it, pt, nl, sv, pl, id, vi, tr, zh, ja, ko, ar, hi, ru, uk, th

## SDK Integration

All SDKs live under `sdk/`:
- `sdk/android/UniversalTranslationSDK/`
- `sdk/ios/UniversalTranslationSDK/`
- `sdk/flutter/universal_translation_sdk/`
- `sdk/react-native/UniversalTranslationSDK/`
- `sdk/web/universal-translation-sdk/`

See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for code examples.

## Documentation

| Topic | Command | Link |
|---|---|---|
| Full setup guide | `./uts docs --open setup` | [SETUP_COMMANDS.md](SETUP_COMMANDS.md) |
| Training guide | `./uts docs --open train` | [docs/TRAINING.md](docs/TRAINING.md) |
| Architecture | `./uts docs --open arch` | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Onboarding | `./uts docs --open sdk` | [docs/ONBOARDING.md](docs/ONBOARDING.md) |
| All docs | `./uts docs --list` | — |

## Component Status

| Component | Status | Notes |
|---|---|---|
| Data Pipeline | ✅ Stable | Download, sample, augment, quality filter, validate |
| Vocabulary System | ✅ Stable | 6 language packs (latin, cjk, arabic, devanagari, cyrillic, thai) |
| Coordinator | ✅ Stable | Load balancing, Redis pool, health monitoring |
| Training | ✅ Implemented | Full model + LoRA, progressive tiers |
| Encoder | ✅ Implemented | 42.7M params, RoPE + SwiGLU |
| Decoder | ✅ Implemented | 108.1M params, cross-attention |
| Android SDK | Scaffolded | Needs encoder binary |
| iOS SDK | Scaffolded | Needs encoder binary |
| Flutter SDK | Scaffolded | Needs encoder binary |
| React Native SDK | Scaffolded | Needs encoder binary |
| Web SDK | Scaffolded | WASM encoder is stub |
| Docker / K8s | ✅ Stable | Docker Compose, Kubernetes, Helm chart |
| Monitoring | ✅ Stable | Prometheus/Grafana dashboards |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

See [docs/ACKNOWLEDGMENTS.md](docs/ACKNOWLEDGMENTS.md).
