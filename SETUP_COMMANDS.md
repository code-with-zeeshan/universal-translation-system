# UTS — Fresh Studio Setup Guide

## Before You Start

**Choose your setup path based on available GPU:**

| Your GPU | Full training? | Expected time (10 epochs) | Budget | Recommended batch | Go to |
|---|---|---|---|---|---|---|
| **A100 40GB / H100** | ✅ Yes | ~6 hours | ~$9.30 | 32 | [Fast Track](#fast-track-a100h100-40gb) |
| **L40s 48GB** | ✅ Yes | ~4 hours | ~$10.00 | 48 | [Fast Track](#fast-track-a100h100-40gb) |
| **L4 24GB** | ✅ Yes | ~10 hours | ~$5.00 | 16 | [Mid-Range L4](#mid-range-l4-24gb) |
| **T4 16GB (Colab / low-cost)** | ⚠️ Marginal | ~20 hours | ~$0 | 8 | [Low-End T4](#low-end-t4-16gb) |
| **No GPU / CPU** | ❌ No | — | — | — | [No GPU / LoRA only](#no-gpu--lora-only) |

---

## Common Steps (All Tiers)

These steps are the same regardless of GPU. Do these first.

### 0. Open a terminal

All commands run in the **terminal** (not notebook).

### 1. Clone

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/code-with-zeeshan/universal-translation-system.git
cd universal-translation-system
```

**Troubleshoot:**
- `git: command not found` → `conda install git -y`
- `Permission denied` → You're outside `/teamspace`. Use `cd /teamspace/studios/this_studio` first

### 2. Install dependencies

```bash
pip install -e ".[train]" 2>&1 | tail -10
```

This installs base + training + serving extras. Takes ~2 min.

**Troubleshoot:**
- `pip: command not found` → `conda install pip -y`
- `error: externally-managed-environment` → Same fix

### 3. Set up environment variables

```bash
# Copy template and edit
cp .env.example .env
# Or set the minimum required:
export UTS_HMAC_KEY="dev-only-change-in-production-1234567890abc"
export UTS_ROLE="master"
export JWT_SECRET="dev-jwt-secret-change-in-production"
```

Add to `~/.bashrc` for permanence. See `docs/environment-variables.md` for all 150+ options.

### 4. Verify environment

```bash
./uts setup --check
```

Expected: GPU detected, Python OK, core imports OK.

### 5. Download & process data (~25 min, any GPU)

```bash
./uts data --pipeline
```

Downloads opus-100, samples, augments, validates. **Auto-resumes by default** if interrupted.

**Key flags:**
- `--force` — re-run from scratch (clears all checkpoints)
- `--no-resume` — skip completed stages instead of resetting
- `--stage sample_filter` — run a single stage
- `--scale 5` — 5× training data targets (for larger training)

**Performance tuning** (in `config/base.yaml` under `data:`, or via CLI flags):
```yaml
data:
  download_max_workers: 8          # parallel downloads (default 4)
  download_parallel_batches: true  # flatten all batches (default false)
  # datasets_cache_dir: /path/to/hf/cache  # HF datasets cache (default: ~/.cache/huggingface/datasets)
```
CLI equivalents: `--download-max-workers 8`, `--download-parallel-batches`, `--datasets-cache-dir /cache/hf`

> **`datasets_cache_dir`**: Sets where HuggingFace `datasets` stores downloaded/processed datasets. Use it to point to a faster disk or a volume with more space. **Not required** — defaults to `~/.cache/huggingface/datasets/` which works fine on most setups. Set it if:
> - Your home partition is small (<50 GB free)
> - You have a dedicated SSD or mounted volume for cache
> - You're running in a container/ephemeral environment that needs a custom path

### 6. Vocabulary packs

Vocabulary packs are **automatically created** by the data pipeline's `create_ready` stage — you typically don't need this step after `uts data --pipeline`.

Use `uts vocab --build` only when you need a **custom build** (e.g., different vocab size, mode, or language groups) without re-running the full pipeline:

```bash
./uts vocab --build --vocab-size 32000
```

This command is **idempotent** — if the config and corpus files haven't changed, it skips immediately.

**Advanced:** Restrict groups or change mode:
```bash
./uts vocab --build --vocab-size 32000 --mode research --groups latin cjk
```

Creates 6 per-script packs at `vocabulary/vocab/`.

---

## Fast Track (A100/H100 40GB+)

**Best experience.** Full model training in ~3 hours ($4.65).

### 7. Train full model

```bash
# Default: 10 epochs (~6h, ~$9.30)
./uts train --full

# Override for fewer epochs for quick test:
./uts train --full --num-epochs 3

# With knowledge distillation (teacher NLLB-200-3.3B):
./uts train --full --distill --distill-alpha 0.5 --distill-temp 4.0
```

**Config already optimized for A100** (`config/base.yaml`):
```yaml
training:
  use_lora: false           # Train all 150.8M params
  num_epochs: 10            # Default (~6h, $9.30). Override: --num-epochs
  lr: 3e-4
  batch_size: 32
  gradient_checkpointing: true
  mixed_precision: true
```

**Auto-resume:** Training saves checkpoints with config hashes. If interrupted, re-run `./uts train --full` to resume. Use `--force` to re-run from scratch. Config changes (epochs, LR, batch size) auto-invalidate the training checkpoint.

**Expected loss trajectory:**
- Epoch 1: ~8.5 → Epoch 3: ~5.5 → Epoch 5: ~4.0 → Epoch 10: ~3.0-3.5

### 8. Evaluate

```bash
# Evaluation test data downloads on demand (no separate download step needed)
./uts eval --model --checkpoint checkpoints/*/best_model.pt
```

**Auto-resume:** Evaluation tracks per-file completion. Re-run to skip already-evaluated files. Use `--force` to re-evaluate all.

Results in `evaluation_reports/`.

---

## Mid-Range (L4 24GB)

Full model training is feasible with smaller batch. Expect ~10 hours.

### 7. Tweak config for L4

Edit `config/base.yaml`:
```yaml
training:
  use_lora: false
  num_epochs: 10              # Default; use --num-epochs to override
  lr: 3e-4
  batch_size: 16              # Reduced from 32
  accumulation_steps: 8       # Keeps effective batch = 128
  gradient_checkpointing: true # Critical for L4 memory
  mixed_precision: true
```

### 8. Train

```bash
./uts train --full --batch-size 16
```

### 9. Evaluate

Same as Fast Track step 8.

---

## Low-End (T4 16GB)

Training the full model is **possible but slow** (~18 hours). You may prefer LoRA-only training for speed.

### Option A: Full training (slow, ~18h)

Edit `config/base.yaml`:
```yaml
training:
  use_lora: false
  num_epochs: 10              # Default; override with --num-epochs
  lr: 3e-4
  batch_size: 8               # Must reduce for 16GB
  accumulation_steps: 16      # Keeps effective batch = 128
  gradient_checkpointing: true
  mixed_precision: true
```

```bash
./uts train --full --batch-size 8
```

### Option B: LoRA adapter training (fast, ~4h)

If you just want to test the system works:

```yaml
training:
  use_lora: true
  lora_r: 16
  lora_r_decoder: 64
  lora_alpha: 32
  num_epochs: 5
  lr: 5e-4
  batch_size: 16
```

```bash
./uts train --full --num-epochs 5
```

Note: LoRA on a randomly initialized backbone gives **poor quality** (BLEU ~0.001). Use only for testing the pipeline, not for production.

---

## No GPU / LoRA Only

You cannot train the full model without a GPU. Your options:

### Option A: Use free Colab T4

1. Upload this repo to Google Drive
2. Open in Colab with T4 runtime (Runtime → Change runtime → T4 GPU)
3. Follow [Low-End T4](#low-end-t4-16gb) steps

### Option B: Train LoRA adapters on Colab

If you already have a trained backbone checkpoint, you can train LoRA adapters on Colab for new languages:

```yaml
training:
  use_lora: true
  lora_r: 16
  lora_r_decoder: 64
  freeze_backbone: true
```

```bash
./uts train --full --checkpoint /path/to/trained_backbone.pt
```

### Option C: Use pre-trained models

Once the community publishes trained checkpoints, you can:
1. Download from Hugging Face Hub
2. Run evaluation only: `./uts eval --model`

---

## Quick Reference

```bash
# Full workflow (copy-paste for A100, ~$8 for full run)
cd /teamspace/studios/this_studio && \
git clone https://github.com/code-with-zeeshan/universal-translation-system.git && \
cd universal-translation-system && \
pip install -e ".[train]" && \
cp .env.example .env && \
. .env && \
./uts data --pipeline && \
./uts vocab --build && \
./uts train --full

# After training is complete (eval data downloads on demand):
./uts eval --model --checkpoint checkpoints/*/best_model.pt

# Monitor training:
./uts tui --config config/base.yaml
```

## Quick Reference — CLI at a Glance

```bash
uts setup --check            # Verify environment readiness
uts setup --verify           # Validate post-deployment setup
uts data --pipeline          # Run full data pipeline (auto-resume)
uts data --pipeline --force  # Re-run from scratch
uts data --pipeline --scale 5  # 5× training data
uts vocab --build            # Build vocabulary packs
uts train --full             # Train (auto-resume, ~3h, $4.65)
uts train --full --force     # Re-run training from scratch
uts train --full --distill   # Knowledge distillation
uts train --full --num-epochs 10  # Override epochs
uts eval --model             # Evaluate (auto-resume, per-file)
uts eval --model --force     # Re-evaluate all files
uts serve --decoder          # Start decoder server
uts serve --coordinator      # Start coordinator
uts publish --repo-id ...    # Publish to HF Hub
uts publish --optimize-decoder  # Quantize + ONNX optimize
uts publish --preflight      # Validate before publishing
uts tools --version          # Check component versions
uts tools --check-compat     # Compatibility checks
uts tools --rotate-secrets   # Rotate API secrets
uts tools --build-encoder    # Build edge encoder ONNX
uts tools --register-decoder # Register decoders
uts docs --open layout       # View filesystem layout docs
```

## File Locations

See `docs/RUNTIME_LAYOUT.md` for the complete filesystem reference (~40 dirs, ~130+ files).

| What | Where |
|---|---|
| Config | `config/base.yaml` |
| Training data | `data/processed/train_final.txt` |
| Vocabulary packs | `vocabulary/vocab/` |
| Model checkpoints | `checkpoints/{experiment}/` |
| Evaluation data | `data/evaluation/{pair}.tsv` |
| Evaluation reports | `evaluation_reports/` |
| Pipeline state (auto-resume) | `.pipeline_state.json`, `.checkpoints/` |
| Pipeline logs | `data/log/data.log` |
| Training logs | `logs/training/` |
| Version config | `version-config.json` |
| Secret config | `.secret_config.yaml` (encrypted) |

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `HMAC key not configured` | Missing `UTS_HMAC_KEY` | `export UTS_HMAC_KEY=...` |
| `JWT secret not configured` | Missing `JWT_SECRET` | `export JWT_SECRET=...` |
| `CUDA out of memory` | Batch too large for your GPU | Reduce `batch_size` or check [your tier](#before-you-start) |
| `No module named '...'` | Missing dependency | `pip install -e ".[train]"` |
| `ConnectionError` | Network issue | Retry; auto-resume picks up where it left off |
| `Checkpoint conflict` | Config hash mismatch | Use `--force` to re-run with new config |
| Loss is NaN | LR too high | Reduce `lr` to `1e-4` |
| BLEU ~0.0 after training | Only 1 epoch completed | Train 5-10 epochs for usable quality |
| BLEU ~0.0 after 10 epochs | LoRA on random init | Set `use_lora: false` for full training |
