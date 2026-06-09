# Onboarding Guide

This document is the reference for the Universal Translation System.
If you haven't yet, run `./uts` — it organizes every tool into workflow groups.

## System Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Data        │ ──>  │  Training        │ ──>  │  Evaluation  │
│  Pipeline    │      │  (encoder +      │      │  (BLEU /     │
│              │      │   decoder)       │      │   COMET)     │
└──────────────┘      └──────────┬───────┘      └──────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Export / Serve │
                        │  (ONNX, server) │
                        └─────────────────┘
```

### Components

| Component | Description | Location |
|---|---|---|
| **Edge Encoder** | 42.7M params, 512d/6L/8H, on-device | `encoder/` |
| **Cloud Decoder** | 108.1M params, 768d/8L/12H, cloud | `cloud_decoder/` |
| **Vocabulary** | 6 per-script SentencePiece packs | `vocabulary/` |
| **Data Pipeline** | Multi-stage download → process | `data/` |
| **Coordinator** | Load-balances decoder pool | `coordinator/` |
| **SDKs** | Android, iOS, Flutter, RN, Web | `sdk/` |

## CLI Reference

The `./uts` command organizes all tools into 8 groups.
Run `./uts <group> --help` for full options.

### `./uts setup` — Environment & Validation

| Flag | Description |
|---|---|
| `--check` | Run GPU/environment readiness check |
| `--config-wizard` | Interactive config file creator |
| `--validate CONFIG` | Validate a config YAML file |
| `--verify` | Verify post-deployment setup (all services reachable, env vars set) |

### `./uts data` — Data Pipeline

| Flag | Description |
|---|---|
| `--pipeline` | Run full data pipeline (download → sample → augment → validate) |
| `--download-only` | Download evaluation test data only (skip training data) |
| `--augment` | Run synthetic data augmentation only |
| `--validate-data` | Validate pipeline output |
| `--domains` | Download domain-specific data for supported domains |
| `--config PATH` | Config file (default: `config/base.yaml`) |
| `--scale N` | Scale training data targets by factor N (e.g., `--scale 5` for 5× data, ~$42 total) |
| `--resume` | Resume from last checkpoint (default behavior) |
| `--no-resume` | Run all stages without checking checkpoint state |
| `--force` | Clear checkpoint and re-run all stages from scratch |
| `--reset` | Reset pipeline state (start fresh) |
| `--stage NAME` | Run a single pipeline stage |

**Pipeline stages (in order):**
1. `download_evaluation` — opus-100 test splits (19 pairs, 2000 each)
2. `download_training` — opus-100 + extra sources
3. `sample_filter` — Deduplicate, filter length/content
4. `augment` — False friends, idioms, backtranslation
5. `create_ready` — Format for training
6. `comet_quality` — Quality filter with COMET
7. `validate` — Validate output data

### `./uts vocab` — Vocabulary Management

> **Note:** The data pipeline already creates vocabulary packs automatically during its `create_ready` stage. You only need `uts vocab --build` for custom builds (different vocab size, mode, or groups). The command is **idempotent** — it skips if the config and corpus files haven't changed.

| Flag | Description |
|---|---|
| `--build` | Build vocabulary packs from processed data |
| `--evolve` | Evolve existing pack (promote frequent unknown tokens) |
| `--vocab-size N` | Tokens per pack (default: 32000) |
| `--mode MODE` | Creation mode: `production`, `research`, `hybrid` |
| `--groups [list]` | Specific packs to build: `latin`, `cjk`, `arabic`, etc. |
| `--pack NAME` | Specific pack to evolve |

### `./uts train` — Model Training

| Flag | Description |
|---|---|
| `--full` | Full model training (all 150.8M params) |
| `--distill` | Knowledge distillation from a teacher model (see TRAINING.md) |
| `--progressive` | Progressive multi-tier training (curriculum) |
| `--lora` | Show LoRA adapter training instructions |
| `--config PATH` | Config file (default: `config/base.yaml`) |
| `--distributed` | Enable distributed (multi-GPU) training |
| `--num-epochs N` | Override number of epochs (default from config) |
| `--batch-size N` | Override batch size |
| `--lr FLOAT` | Override learning rate |
| `--experiment-name NAME` | Experiment name for logging |
| `--checkpoint PATH` | Resume training from checkpoint |
| `--force` | Ignore training checkpoint, re-train from scratch |
| `--start-tier TIER` | Progressive: start from specific tier (`tier1`–`tier4`) |
| `--validate-final` | Progressive: run validation on final model |
| `--teacher MODEL` | Teacher model for distillation (default: `facebook/nllb-200-3.3B`) |
| `--distill-alpha FLOAT` | CE vs KD loss weight, 0–1 (default: `0.5`) |
| `--distill-temp FLOAT` | Distillation temperature (default: `4.0`) |

**Config reference (key settings in `config/base.yaml`):**

```yaml
model:
  max_vocab_size: 32000          # Embedding table size
  hidden_dim: 512                 # Encoder dimension
  num_layers: 6                   # Encoder layers
  decoder_dim: 768                # Decoder dimension
  decoder_layers: 8               # Decoder layers

training:
  use_lora: false                 # false = train full model
  num_epochs: 5                   # Default epochs (override: --num-epochs 10)
  lr: 3e-4                        # Learning rate
  warmup_steps: 1000              # LR warmup
  batch_size: 32                  # Per-GPU batch
  gradient_checkpointing: true    # Memory savings
  mixed_precision: true           # bfloat16 training

data:
  active_languages: [en, es, fr, de, it, pt, nl, sv, pl, id, vi, tr,
                     zh, ja, ko, ar, hi, ru, uk, th]
```

### `./uts eval` — Evaluation & Benchmarking

| Flag | Description |
|---|---|
| `--model` | Evaluate a trained model checkpoint |
| `--download` | Download evaluation test data |
| `--benchmark` | Benchmark model performance (latency, throughput) |
| `--config PATH` | Config file (default: `config/base.yaml`) |
| `--checkpoint PATH` | Path to model checkpoint (`.pt` file) |
| `--test-data PATH` | Test data directory (default: `data/evaluation`) |
| `--profile-steps N` | Steps to profile (default: 10) |
| `--output-dir PATH` | Profiling output directory (default: `profiling`) |
| `--force` | Re-evaluate all files even if previously completed |

### `./uts serve` — Serving Infrastructure

| Flag | Description |
|---|---|
| `--decoder` | Start cloud decoder server (FastAPI/LitServe) |
| `--coordinator` | Start coordinator service (load balancer) |
| `--setup` | Configure serving infrastructure (Docker, etc.) |
| `--all` | Setup all serving components |
| `--redis MODE` | Manage Redis: `install`, `start`, `stop`, `status` |

### `./uts tools` — Utilities

| Flag | Description |
|---|---|
| `--validate-config PATH` | Validate config YAML |
| `--check-references` | Check referenced files exist |
| `--check-consistency` | Check internal consistency |
| `--suggest` | Suggest config improvements |
| `--verbose` | Verbose validation output |
| `--check-gpu` | Run GPU readiness check |
| `--prefetch` | Prefetch artifacts from Hugging Face Hub |
| `--pairs [list]` | Language pairs for prefetch (e.g., `en:es en:fr`) |
| `--packs [list]` | Vocab packs for prefetch (e.g., `latin cjk`) |
| `--repo-id ID` | Hugging Face Hub repository ID |
| `--rotate-secrets` | Rotate JWT secrets |
| `--key-type TYPE` | Key type for rotation (e.g., `hmac`, `jwt`) |
| `--set-env FILE` | Set environment variables from a template file |
| `--upload [REPO]` | Upload artifacts to Hugging Face Hub |
| `--register-decoder` | Register a decoder node with the coordinator |
| `--build-encoder` | Build the encoder core for edge deployment |
| `--build-target TARGET` | Build target (e.g., `wasm`, `android`, `ios`) |
| `--check-compat` | Check decoder node compatibility with current coordinator |
| `--version` | Show component versions |
| `--version-config PATH` | Show version info for specific config file |
| `--install` | Install system dependencies |
| `--train` | Install training dependencies |
| `--serve` | Install serving dependencies |

### `./uts docs` — Documentation Browser

| Flag | Description |
|---|---|
| `--open TOPIC` | Open documentation for a topic |
| `--list` | List available documentation topics |

Available topics: `setup`, `train`, `arch`, `vocab`, `deploy`, `api`, `env`, `sdk`, `monitor`, `faq`, `trouble`, `roadmap`, `vision`, `layout`, `version`, `secret`, `tui`, `publish`, `test`

## Language Expansion Strategy (Adding Languages Beyond 20)

The system is designed for zero-full-retraining language expansion:

1. **Current phase** — Train full backbone on 20 languages (`use_lora: false`)
2. **New language phase** — When adding language #21+:
   - Set `use_lora: true` in config
   - Add language code to `data.languages`
   - Train only LoRA adapters + target language adapter (~7M params)
   - This preserves the backbone while adding capacity

```bash
# To train LoRA adapters for new languages:
# 1. Update config: add language to active_languages, set use_lora: true
# 2. Run:
./uts train --full --experiment-name "lang-21-adapter"
```
