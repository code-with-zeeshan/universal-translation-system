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
| **Edge Encoder** | 42.7M params, 512d/6L/8H, on-device | `runtime/encoder/` |
| **Cloud Decoder** | 108.1M params, 768d/8L/12H, cloud | `runtime/cloud_decoder/` |
| **Vocabulary** | 6 per-script SentencePiece packs | `vocabulary/` |
| **Data Pipeline** | Multi-stage download → process | `pipeline/data/` |
| **Coordinator** | Load-balances decoder pool | `runtime/coordinator/` |
| **SDKs** | Android, iOS, Flutter, RN, Web | `sdk/` |

## CLI Reference

The `./uts` command organizes all tools into 10 groups.
Run `./uts <group> --help` for full options.

### Quick Decision Guide

| Role | When you want to... | Run this | Then... |
|---|---|---|---|
| **Any** | Get started from scratch | See [GETTING_STARTED.md](GETTING_STARTED.md) | Two paths: Builder or Consumer |
| **Any** | Set up a new machine/studio | `./uts setup --check` | Tweak config per [SETUP_COMMANDS.md](../SETUP_COMMANDS.md) by GPU tier |
| **Any** | Validate your config file | `./uts tools --validate-config <path>` | Add `--check-references`, `--check-consistency` |
| **Any** | Check installed dependencies | `./uts setup --check-deps` (or `tools --check-deps`) | Run `pip install -e ".[train]"` if missing |
| **Any** | Browse docs by topic | `./uts docs --list` | Then `./uts docs --open <topic>` |
| **Builder** | Download & process training data | `./uts data --pipeline` | Add `--scale 5` for more data |
| **Builder** | Build vocabulary packs (custom) | `./uts vocab --build` | Pipeline auto-builds; only run this for custom configs |
| **Builder** | Train a full model | `./uts train --full` | Add `--num-epochs`, `--batch-size`, or `--distill` |
| **Builder** | Resume interrupted training | `./uts train --full` (same command) | Add `--force` to restart from scratch |
| **Builder** | Train with knowledge distillation | `./uts train --distill` | Set `--teacher`, `--distill-alpha`, `--distill-temp` |
| **Builder** | Train progressively (curriculum) | `./uts train --progressive` | Start from a specific `--start-tier` |
| **Builder** | Evaluate a trained model | `./uts eval --model --checkpoint <path>` | Add `--benchmark` for latency/throughput |
| **Builder** | Benchmark model performance | `./uts eval --benchmark` | Results in `evaluation_reports/` |
| **Builder** | Convert model to ONNX/CoreML/TFLite | `./uts tools --convert-task onnx` | Requires `--convert-model-path` |
| **Builder** | End-to-end build + upload to HF Hub | `./uts tools --build-and-upload <repo>` | Add `--create-vocabs`, `--convert-models` |
| **Builder** | Add a new language (#21+) | Phase 1 backbone must exist. Then: `./uts train --lora --experiment-name "lang-21-adapter" --num-epochs 5` | LoRA mode is auto-forced by `--lora`; no need to edit config. `--experiment-name` is just a log label. 1 epoch = BLEU ~0. |
| **Consumer** | Download pre-built artifacts from HF Hub | `./uts tools --prefetch --repo-id <id>` | Artifacts go to `models/production/`, `vocabs/`, `adapters/` |
| **Consumer** | Evaluate without training | `./uts eval --model --checkpoint models/production/best_model.pt` | Eval data downloads automatically |
| **Consumer** | Serve the decoder locally | `./uts serve --decoder` | Then `./uts serve --coordinator` for load balancing |
| **Consumer** | Integrate via SDKs | See [SDK_INTEGRATION.md](SDK_INTEGRATION.md) | Android, iOS, Flutter, RN, Web |
| **Publish** | Run preflight checks | `./uts publish --preflight` | Validates data, vocabs, models before publishing |
| **Publish** | Optimize decoder for deployment | `./uts publish --optimize-decoder` | Quantize + ONNX optimize (standalone) |
| **Publish** | Publish model to Hugging Face Hub | `./uts publish --repo-id your-org/model` | Use `--no-onnx` or `--no-quantize` to skip steps |
| **Publish** | Upload existing artifacts only | `./uts publish --upload-only` | Skips ONNX export and quantization |
| **Ops** | Deploy decoder server | `./uts serve --decoder` | Needs `decoder.pt` + vocab packs in place |
| **Ops** | Deploy coordinator (load balancer) | `./uts serve --coordinator` | See [DECODER_POOL.md](DECODER_POOL.md) for pool setup |
| **Ops** | Configure serving infrastructure | `./uts serve --setup` | Docker, dependencies |
| **Ops** | Manage Redis | `./uts serve --redis install\|start\|stop\|status` | Required for distributed decoder pool |
| **Ops** | Check API version compatibility | `./uts serve --check-api-versions` | Requires `--coordinator-url` + `--decoder-url` |
| **Ops** | Rotate API secrets | `./uts tools --rotate-secrets` | Use `--key-name <name>` for specific secret |
| **Ops** | Register a decoder node | `./uts tools --register-decoder` | Adds node to coordinator pool |
| **Ops** | Decoder node compatibility check | `./uts tools --check-compat` | Validates API/schema/ONNX versions |
| **Ops** | Check component versions | `./uts tools --version` | Or `--version-config <path>` for a specific config |
| **Ops** | Update API schema hash | `./uts tools --update-schema-hash` | Recomputes and updates `version-config.json` |
| **Ops** | Build encoder C++ core for edge | `./uts tools --build-encoder` | Targets: Linux, macOS, Android, iOS |
| **Ops** | Verify post-deployment setup | `./uts setup --verify` | Checks services, endpoints, sample translation |
| **Dev** | Run the TUI dashboard | `./uts tui` | Or `--pipeline` / `--train` for focused views |
| **Dev** | Run tests | See [TESTING.md](TESTING.md) | Unit, integration, security tests |
| **Dev** | GPU readiness check | `./uts setup --check` (or `tools --check-gpu`) | Validates CUDA, PyTorch, nvidia-smi |

### `./uts setup` — Environment & Validation

| Flag | Description |
|---|---|
| `--check` | Run GPU/environment readiness check |
| `--config-wizard` | Interactive config file creator |
| `--validate CONFIG` | Validate a config YAML file |
| `--verify` | Verify post-deployment setup (all services reachable, env vars set) |
| `--check-deps` | Check installed dependencies against requirements files |

### `./uts data` — Data Pipeline

| Flag | Description |
|---|---|
| `--pipeline` | Run data pipeline (download → sample → augment → create → validate → vocab). Eval data not included; use `uts eval --download` |
| `--interactive` | Interactive stage selector (TUI wizard) — toggle stages on/off with dynamic time estimates, generates config override |
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

**Pipeline stages (in order, default set):**
1. `download_training` — OPUS-100 + extra sources
2. `sample_filter` — Deduplicate, filter length/content
3. `augment` — False friends, idioms, backtranslation, pivots
4. `create_ready` — Merge all sources into train_final.txt / val_final.txt
5. `validate` — Validate output data and vocabulary files
6. `vocabulary` — Build vocabulary packs from monolingual corpora

**Optional heavy stages** (enable via `config.pipeline.enabled_stages`):
- `wikipedia_backtranslation` — Download Wikipedia monolingual data for backtranslation (~200K sentences/lang)
- `direct_opus` — Direct OPUS.nlpl.eu fallback download for missing pairs
- `knowledge_distillation` — Distill NLLB-3.3B teacher into training data (GPU required)
- `download_evaluation` — Pre-fetch evaluation test sets (use `uts eval --download` instead)
- `comet_quality` — Neural quality filter with Unbabel/wmt22-comet-da (GPU + 12GB+ VRAM)

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
| `--full` | Run training pipeline (reads `use_lora` from config — all 150.8M or LoRA depending on config) |
| `--distill` | Knowledge distillation from a teacher model (see TRAINING.md) |
| `--progressive` | Progressive multi-tier training (curriculum) |
| `--lora` | LoRA adapter training (forces `use_lora=true`, overrides config) |
| `--config PATH` | Config file (default: `config/base.yaml`) |
| `--distributed` | Enable distributed (multi-GPU) training |
| `--num-epochs N` | Override number of epochs (default from config) |
| `--batch-size N` | Override batch size |
| `--lr FLOAT` | Override learning rate |
| `--experiment-name NAME` | Experiment name for logging (does NOT change training mode) |
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
  num_epochs: 10                  # Default epochs (override: --num-epochs)
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
| `--check-api-versions` | Check runtime API version compatibility (requires `--coordinator-url` + `--decoder-url`) |
| `--coordinator-url URL` | Coordinator URL for API version check |
| `--decoder-url URL` | Decoder URL for API version check |

### `./uts publish` — Model Publishing

Publish a trained model to Hugging Face Hub with optional ONNX export and quantization.

| Flag | Description |
|---|---|
| `--repo-id ID` | **Required for publish:** HF Hub repo ID (e.g., `your-org/universal-translation-system`) |
| `--checkpoint PATH` | Path to trained model checkpoint |
| `--no-onnx` | Skip ONNX export |
| `--no-quantize` | Skip quantization |
| `--upload-only` | Upload existing artifacts without rebuilding |
| `--preflight` | **[standalone]** Run preflight checks (exits after check, no publish) |
| `--optimize-decoder` | **[standalone]** Quantize + optimize decoder (exits after, no publish) |

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for details.

### `./uts tui` — Terminal Dashboard

| Flag | Description |
|---|---|
| *(none)* | Open combined pipeline + training view (default) |
| `--pipeline` | Pipeline progress view only |
| `--train` | Training metrics view only |

See [docs/TUI.md](docs/TUI.md) for keyboard shortcuts and layout.

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
| `--key-type TYPE` | Key type for rotation (`hs256`, `rs256`, `all`) |
| `--key-name NAME` | Specific secret key to rotate (e.g., `coordinator_jwt_secret`) |
| `--kid KID` | Key ID for RS256 rotation |
| `--set-env` | Set rotated secrets in current env (only with `--rotate-secrets`) |
| `--upload [REPO]` | Upload artifacts to Hugging Face Hub |
| `--register-decoder` | Register a decoder node with the coordinator |
| `--build-encoder` | Build the encoder core for edge deployment |
| `--build-target TARGET` | Build target (e.g., `wasm`, `android`, `ios`) |
| `--check-compat` | Check decoder node compatibility with current coordinator |
| `--version` | Show component versions |
| `--version-config PATH` | Show version-config.json contents (standalone) or pass to `--check-compat` |
| `--check-deps` | Check installed dependencies against requirements (same as `setup --check-deps`) |
| `--update-schema-hash` | Recompute API schema hash and update version-config.json |
| `--convert-task TASK` | Model conversion task (`onnx`, `coreml`, `tflite`, `tensorrt`) |
| `--convert-model-path PATH` | Source model for conversion (required with `--convert-task`) |
| `--convert-onnx-path PATH` | Target ONNX path for conversion |
| `--convert-opset N` | ONNX opset version (default: 17) |
| `--build-and-upload [REPO]` | End-to-end: create vocabs, convert models, upload to HF Hub |
| `--create-vocabs` | Create vocab packs (with `--build-and-upload`) |
| `--convert-models` | Convert models to ONNX (with `--build-and-upload`) |
| `--vocab-groups [list]` | Vocab groups for `--build-and-upload` |
| `--install` | Install system dependencies |
| `--train` | Install training dependencies |
| `--serve` | Install serving dependencies |

### `./uts docs` — Documentation Browser

| Flag | Description |
|---|---|
| `--open TOPIC` | Open documentation for a topic |
| `--list` | List available documentation topics |

Available topics: `start`, `setup`, `train`, `arch`, `vocab`, `deploy`, `api`, `env`, `sdk`, `monitor`, `faq`, `trouble`, `vision`, `layout`, `version`, `secret`, `tui`, `publish`, `test`, `ci`, `ci_build`, `decoder_pool`, `redis`, `autoscale`, `new_lang`, `security`, `sdk_publish`, `perf`

Run `./uts docs --list` to see the latest list with descriptions.

## Language Expansion Strategy (Adding Languages Beyond 20)

The system is designed for zero-full-retraining language expansion, **but a fully-trained Phase 1 backbone must exist**:

1. **Phase 1 (required)** — Train full backbone on 20 languages (`use_lora: false`). This trains all 150.8M params and must exist before LoRA produces usable results. You can get a Phase 1 backbone in two ways:
   - **Train it yourself** — Run `uts train --full` (10 epochs, ~$9 on A100)
   - **Download a community-published one** — `uts tools --prefetch --repo-id <org>/universal-translation-system` (free, if someone has published)

2. **Phase 2 (add languages)** — Once you have a trained backbone (yours or downloaded), add language #21+:
   - Set `use_lora: true` in config
   - Add language code to `data.languages`
   - Train only LoRA adapters + target language adapter (~7M params)
   - **Train 5–10 LoRA epochs minimum** — 1 epoch still yields BLEU ~0

```bash
# Option A: Train Phase 1 from scratch (takes ~6h on A100)
./uts data --pipeline
./uts train --full

# Option B: Download a completed Phase 1 backbone (free, if available)
./uts tools --prefetch --repo-id your-org/universal-translation-system

# Then train LoRA adapters for new languages:
# (edit config: add language to active_languages, set use_lora: true)
./uts train --full --experiment-name "lang-21-adapter" --num-epochs 5
```

**Why BLEU is ~0 with LoRA?** Three common causes:
- Backbone was never fully trained (only bootstrapped or randomly initialized — downloading a trained checkpoint fixes this)
- Only 1 epoch of LoRA training (needs 5–10)
- LoRA on a randomly initialized model (needs the full 150.8M backbone trained first)
