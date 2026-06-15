# Getting Started

This repo supports **two distinct user paths**. Choose the one that matches your goal:

| You want to... | Follow |
|---|---|
| Clone and build everything from scratch (data pipeline → train → eval) on a fresh Lightning AI studio | **Path A — Builder** |
| Use already-trained models, vocabulary packs, and SDKs without training | **Path B — Consumer** |

---

## Path A: Build from Scratch

**You have a GPU (T4, L4, A100, L40s, H100) and want to train your own model.**

### 1. Clone & install

```bash
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system
pip install -e ".[train]"
# Note: ".[train]" installs only base.txt + train.txt.
# For all features (serve, decoder, export, tui): pip install -e ".[all]"
cp .env.example .env
export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"
export JWT_SECRET="dev-jwt-secret-change-in-production"
```

### 2. Verify environment

```bash
./uts setup --check
```

Expected: GPU detected, Python version OK, all imports resolve.

### 3. Run the data pipeline

Downloads opus-100, samples, augments, validates, builds vocab. Auto-resumes if interrupted.

```bash
./uts data --pipeline
```

- `--scale 5` — 5 training data for a larger model
- `--force` — re-run from scratch
- `--stage sample_filter` — run a single stage

### 4. Vocabulary packs

The pipeline already builds them automatically. Only run this for custom builds:

```bash
./uts vocab --build --vocab-size 32000
```

### 5. Train

```bash
./uts train --full
```

**GPU-specific notes:** See [SETUP_COMMANDS.md](../SETUP_COMMANDS.md) for the exact `--batch-size` and config tweaks per GPU tier. The table below is a quick reference:

| GPU | Recommended batch | Approx time (10 epochs) |
|---|---|---|
| A100 40GB / H100 | 32 | ~6h / ~4h |
| L40s 48GB | 48 | ~5h |
| L4 24GB | 16 | ~10h |
| T4 16GB | 8 | ~20h |

Auto-resume: re-run same command after interruption. Use `--force` to restart.

### 6. Evaluate

```bash
./uts eval --model --checkpoint checkpoints/*/best_model.pt
```

Evaluation data downloads on demand. Results in `evaluation_reports/`.

### 7. (Optional) Publish to Hugging Face Hub

```bash
./uts publish --repo-id your-org/universal-translation-system
```

Splits checkpoint → exports ONNX → quantizes → uploads. See [PUBLISHING.md](PUBLISHING.md).

---

## Path B: Consumer

**You already have a trained model (yours or a community release) and want to use it.**

### Option 1: Download pre-built artifacts from HF Hub

If someone has published to Hugging Face Hub:

```bash
# Download model, vocabs, and adapters
./uts tools --prefetch --repo-id your-org/universal-translation-system
```

Or manually download from: `https://huggingface.co/your-org/universal-translation-system`

The repo contains:
```
models/production/encoder.pt       # 42.7M param encoder
models/production/decoder.pt       # 108.1M param decoder
models/production/encoder.onnx     # ONNX-exported encoder
vocabs/*.msgpack                   # 6 per-script vocabulary packs
adapters/*.pt                      # Language-specific LoRA adapters
```

Place files in the expected locations (run `./uts tools --check-references` to verify):

```bash
mkdir -p models/production vocabs
# Copy encoder.pt, decoder.pt → models/production/
# Copy *.msgpack → vocabs/
```

### Option 2: Evaluate without training

```bash
# Skip data pipeline, skip training — go straight to eval
./uts eval --model --checkpoint models/production/best_model.pt
```

Evaluation test data downloads automatically on first run.

### Option 3: Serve the decoder locally

Run the decoder server (needs `decoder.pt` + vocab packs in place):

```bash
# Start decoder (one terminal)
./uts serve --decoder

# Start coordinator (another terminal)
./uts serve --coordinator
```

See [API.md](API.md) for endpoints (`/decode`, `/health`, etc.) and auth details.

### Option 4: Integrate via SDKs

All SDKs live under `sdk/` and connect to a running coordinator/decoder:

| Platform | Location | Quick start |
|---|---|---|
| Android | `sdk/android/UniversalTranslationSDK/` | See [SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) |
| iOS | `sdk/ios/UniversalTranslationSDK/` | See [SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) |
| Flutter | `sdk/flutter/universal_translation_sdk/` | See [SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) |
| React Native | `sdk/react-native/UniversalTranslationSDK/` | See [SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) |
| Web | `sdk/web/universal-translation-sdk/` | See [SDK_INTEGRATION.md](../docs/SDK_INTEGRATION.md) |

Each SDK handles:
- **Edge encoding** (where supported) — encodes text into embeddings on-device
- **Coordinator-aware routing** — sends encoded embeddings to least-loaded decoder
- **Local decoder auto-discovery** — finds decoders on the same LAN via mDNS
- **Vocabulary loading** — downloads/loads the right vocab pack per language pair

To publish your own SDK builds for distribution, see [SDK_PUBLISHING.md](SDK_PUBLISHING.md).

### Option 5: Deploy to production

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose, Kubernetes, and Helm:

```bash
# Quick Docker Compose (decoder + coordinator + Redis)
docker compose up -d
```

---

## Which path should I choose?

See the [Quick Decision Guide](ONBOARDING.md#quick-decision-guide) in ONBOARDING.md for a complete role-based reference (40+ scenarios across Builder, Consumer, Publisher, Ops, and Dev workflows).

---

## Troubleshooting

See [TROUBLESHOOT.md](TROUBLESHOOT.md) or [SETUP_COMMANDS.md](../SETUP_COMMANDS.md#common-issues) for common errors and fixes.

| Error | Likely cause | Fix |
|---|---|---|
| `No module named '...'` | Missing install | `pip install -e ".[train]"` |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` or increase `accumulation_steps` in config |
| `HMAC key not configured` | Missing env var | `export UTS_HMAC_KEY=...` |
| `Checkpoint conflict` | Config hash mismatch | `./uts train --full --force` |
| `BLEU ~0.0` | Only 1 epoch or LoRA on random init | Train 10 epochs or set `use_lora: false` |
