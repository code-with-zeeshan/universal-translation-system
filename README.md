# Universal Translation System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A flexible and scalable translation platform supporting 20 languages via an edge-cloud architecture. A 42.7M-parameter encoder runs on-device (SDK), compressing text into embeddings; a 108.1M-parameter decoder in the cloud produces translations. A smart coordinator routes requests across a pool of decoder nodes.

> **New**: Terminal UI dashboard — `uts tui` to monitor pipeline + training in real time.

## Key Innovation

Rather than bundling a huge model per language, the system splits the workflow for maximum efficiency and scalability:
- **Edge Universal Encoder (42.7M params)**: 512-dim, 6-layer, 8-head encoder with RoPE + SwiGLU for on-device inference.
- **On-demand Vocabulary Packs (2–4MB)**: 32K BPE vocabulary via SentencePiece, grouped by script (latin, cjk, cyrillic, arabic, devanagari, thai).
- **Cloud Decoder (108.1M params)**: 768-dim, 8-layer, 12-head cross-attention decoder, served via FastAPI/LitServe.
- **Dual-Phase Training**: Phase 1 trains all 150.8M params for strong multilingual representations. Phase 2 freezes the backbone and trains only LoRA adapters (~7M params) when adding new languages — no full retraining needed. Knowledge distillation supported (`--distill`).
- **Smart Coordinator**: Routes to least-loaded decoders, circuit breaker, elastic scaling, 50ms batch window.
- **Auto-Resume Pipeline**: Config-hash checkpointing across data→train→eval. Crashes resume from last completed step.
- **Multi-SDK**: Native Android/iOS/Flutter/React Native/Web SDKs under `sdk/` with coordinator-aware routing + local decoder preference.

## Features

- 20 language support with dynamic vocabulary loading (32K BPE, SentencePiece)
- Edge encoding, cloud decoding architecture
- Native SDKs for Android, iOS, Flutter, React Native, Web — coordinator-aware, local decoder auto-discovery
- **Dual-phase training**: Full backbone → LoRA adapters + **knowledge distillation**
- **Auto-resume pipeline**: Checkpointing with config-hash invalidation across data→train→eval
- **Terminal UI dashboard**: `uts tui` for real-time pipeline + training + GPU monitoring
- Full-system monitoring with Prometheus/Grafana
- Docker Compose, Kubernetes, Helm chart deployment
- Redis integration for distributed decoder pool management
- mDNS auto-discovery for decoder nodes
- Secret management (keyring + encrypted file + file-based bootstrap)
- Centralized version management with semver compatibility checks
- ONNX export + quantization pipeline for edge deployment

## Getting Started

Choose your path:

| You want to... | Start here |
|---|---|
| Clone on a fresh GPU machine and train from scratch | [docs/GETTING_STARTED.md — Path A: Builder](docs/GETTING_STARTED.md) |
| Use pre-trained models, vocabs, and SDKs without training | [docs/GETTING_STARTED.md — Path B: Consumer](docs/GETTING_STARTED.md) |

Quick reference for builders:

```bash
# Clone & install
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system
cp .env.example .env              # Edit: set UTS_HMAC_KEY, HF_TOKEN, etc.
pip install -e ".[train]"

# Full pipeline (auto-resumes if interrupted)
./uts data --pipeline             # ~30 min
./uts train --full                # ~6h on A100 for 10 epochs
./uts eval --model                # Evaluate all language pairs
```

See [SETUP_COMMANDS.md](SETUP_COMMANDS.md) for GPU-tier-specific batch sizes and config.

## Training Duration

Default is **10 epochs (~6h on A100, $9.30)**. Use `--num-epochs` to adjust.

| GPU | Default (10 epochs) | 5 epochs (quick test) | Notes |
|---|---|---|---|
| A100 40GB | ~6h ($9.30) | ~3h ($4.65) | Sweet spot |
| H100 | ~4h | ~2h | Fastest |
| L4 24GB | ~10h | ~5h | Viable with gradient checkpointing |
| T4 16GB | ~20h | ~10h | Needs gradient checkpointing |

**Auto-resume**: If interrupted, `uts data --pipeline` and `uts train --full` pick up where you left off. Use `--force` to re-run from scratch.

See [SETUP_COMMANDS.md](SETUP_COMMANDS.md) for exact commands per GPU tier.

## Unified CLI: `./uts`

All tools are organized into 10 workflow groups. Run `./uts <group> --help` for details:

| Group | Purpose |
|---|---|
| `./uts setup` | Environment check, config wizard, validate config, verify deployment |
| `./uts data` | Download, sample, augment, validate pipeline (auto-resume, `--force`, `--scale`) |
| `./uts vocab` | Build or evolve vocabulary packs |
| `./uts train` | Full model / distillation / progressive / LoRA training (auto-resume, `--force`) |
| `./uts eval` | Evaluate model, benchmark, download test data (per-file checkpoint, `--force`) |
| `./uts publish` | Publish model to HF Hub (split, ONNX, quantize, upload) |
| `./uts serve` | Start decoder server, coordinator, Redis |
| `./uts tui` | Terminal UI dashboard for live pipeline + training + GPU monitoring |
| `./uts tools` | Config validation, GPU check, secrets rotation, prefetch, version, compatibility |
| `./uts docs` | Open documentation by topic (`--list` for 28 topics) |

## Language Expansion Strategy

The system is designed for zero-full-retraining language expansion:

1. **Phase 1 (current):** Train full backbone on 20 languages (`use_lora: false` in config). Builds strong multilingual representations.
2. **Phase 2 (future):** Freeze backbone, set `use_lora: true`, train LoRA adapters + target language adapter for languages 21+. Only ~7M trainable params vs 150.8M.

```bash
# Phase 1: Full model training
./uts train --full                    # Default 10 epochs (~6h)
./uts train --full --num-epochs 5     # Quick test (~3h)

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

Run `./uts docs --list` for all 28 topics. Key ones:

| Topic | Command | Covers |
|---|---|---|
| Getting Started | `uts docs --open start` | Builder (scratch) vs Consumer (pre-built) paths |
| Onboarding | `uts docs --open setup` | Full CLI reference, config, language expansion |
| Training | `uts docs --open train` | Strategy, distillation, memory, monitoring |
| Architecture | `uts docs --open arch` | System design, model specs, data flow |
| API | `uts docs --open api` | Decoder + coordinator endpoints, auth |
| Deployment | `uts docs --open deploy` | Docker Compose, K8s, Helm |
| TUI Dashboard | `uts docs --open tui` | Live pipeline/training monitoring |
| Publishing | `uts docs --open publish` | HF Hub split → ONNX → quantize → upload |
| Version Mgmt | `uts docs --open version` | Semver, compatibility, release |
| Secret Mgmt | `uts docs --open secret` | Bootstrap, credential manager, rotation |
| Runtime Layout | `uts docs --open layout` | Every file/directory created during operation |
| Testing | `uts docs --open test` | Test suite reference |
| Environment Vars | `uts docs --open env` | All 150+ env vars with defaults |

## Component Status

| Component | Status | Notes |
|---|---|---|
| Data Pipeline | ✅ Stable | Auto-resume, config-hash invalidation, per-pair sub-stage tracking |
| Vocabulary System | ✅ Stable | 6 language packs (latin, cjk, arabic, devanagari, cyrillic, thai) |
| Training | ✅ Stable | Full model, knowledge distillation, progressive, LoRA, auto-resume |
| Evaluation | ✅ Stable | Per-file checkpoint, auto-resume, BLEU/COMET/chrF |
| Coordinator | ✅ Stable | Circuit breaker, 50ms batcher, Redis/etcd pool, mDNS |
| Encoder | ✅ Implemented | 42.7M params, RoPE + SwiGLU |
| Decoder | ✅ Implemented | 108.1M params, cross-attention |
| TUI Dashboard | ✅ Stable | Real-time pipeline + training + GPU monitoring |
| Publishing | ✅ Stable | Split → ONNX → quantize → HF Hub upload |
| Version Management | ✅ Stable | Centralized semver, compatibility checks, CI gate |
| Secret Management | ✅ Stable | 3-layer: bootstrap → credential manager → secure serialization |
| Auto-Resume Pipeline | ✅ Stable | Cross-stage data→train→eval with config-hash invalidation |
| Android SDK | ✅ Enhanced | Coordinator-aware, local decoder + port auto-scan |
| iOS SDK | ✅ Enhanced | Coordinator-aware, local decoder + port auto-scan |
| Flutter SDK | ✅ Enhanced | Coordinator-aware, local decoder preference |
| React Native SDK | ✅ Enhanced | Coordinator-aware, local decoder preference |
| Web SDK | ✅ Enhanced | Coordinator-aware, WASM encoder |
| Docker / K8s | ✅ Stable | Docker Compose, Kubernetes, Helm chart |
| Monitoring | ✅ Stable | Prometheus/Grafana dashboards |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

See [docs/ACKNOWLEDGMENTS.md](docs/ACKNOWLEDGMENTS.md).
