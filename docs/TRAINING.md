# Training Guide

Train a 150.8M-parameter multilingual NMT model (encoder 42.7M + decoder 108.1M).

## Quick Start

```bash
./uts train --full --config config/base.yaml
```

## Training Strategy

### Phase 1: Full Backbone Training (current)

Train all 150.8M parameters on 20 languages. This builds strong multilingual representations.

| Setting | Value | Why |
|---|---|---|---|
| `use_lora` | `false` | Train ALL params, not just adapters |
| `num_epochs` | 10 (default) | Override with `--num-epochs` for shorter/longer training |
| `lr` | `3e-4` | Standard for full model training |
| `warmup_steps` | 1000 | Quick warmup to target LR |
| `batch_size` | auto-probed | Determined at startup by `DynamicBatchSizer.probe()` — dummy forward/backward at increasing sizes until OOM, then steps back one size for headroom. Typical results: A100 40GB → 64–96, A100 80GB → 128–192, H100 → 256+ |
| `accumulation_steps` | 4 | Effective batch = 128 |

**Expected loss trajectory (default 10 epochs):**
- Epoch 0: ~11.0 (random init)
- Epoch 1: ~8.5
- Epoch 3: ~5.5
- Epoch 5: ~4.0
- Epoch 10: ~3.0-3.5

**Expected BLEU after 10 epochs:** 15-25 (varies by language pair)

### Phase 2: Knowledge Distillation

Transfer knowledge from a teacher model (NLLB-200-distilled-1.3B by default) to the student model via KL-divergence loss. The same 1.3B model is shared with the data augmentation pipeline, avoiding the need to load two separate NLLB variants.

**Loss formula:** `alpha * CE(student, hard_labels) + (1-alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))`

```bash
# Distill from NLLB-200-1.3B (default teacher)
uts train --distill

# Custom teacher checkpoint
uts train --distill --teacher /path/to/teacher_checkpoint.pt

# Adjust distillation parameters
uts train --distill --distill-alpha 0.3 --distill-temp 5.0 --num-epochs 10
```

| Flag | Default | Description |
|---|---|---|
| `--distill` | — | Enable knowledge distillation training |
| `--teacher` | `facebook/nllb-200-distilled-1.3B` | Teacher model (HF model ID or local checkpoint path) |
| `--distill-alpha` | `0.5` | CE vs KD loss weight (0 = pure KD, 1 = pure CE) |
| `--distill-temp` | `4.0` | Softmax temperature for KD. Higher = softer distribution |

**When to use distillation:**
- You want soft targets that preserve semantic nuance better than hard labels
- Domain adaptation: fine-tune student on in-domain data with a general-domain teacher
- The 1.3B teacher fits on T4 GPUs (~6GB VRAM), making distillation feasible on modest hardware
- Default `--distill-alpha 0.5` balances student CE loss with teacher guidance

**Teacher loading priority:**
1. Local checkpoint path (if provided via `--teacher`)
2. NLLB-200-distilled-1.3B from Hugging Face (if CUDA available and transformers installed)
3. Falls back to CE-only training if no teacher can be loaded

### Phase 3: LoRA Adapter Training (future languages)

When adding language #21+, freeze backbone and train adapters:

```yaml
training:
  use_lora: true
  lora_r: 16        # Encoder LoRA rank
  lora_r_decoder: 64 # Decoder LoRA rank (cloud can handle larger)
  lora_alpha: 32
  lora_dropout: 0.05
```

## CLI Flags

```bash
./uts train --full --config config/base.yaml \
  --num-epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --experiment-name my-run
```

| Flag | Default | Description |
|---|---|---|
| `--full` | — | Run training pipeline (reads `use_lora` from config — all 150.8M or LoRA depending on config) |
| `--distill` | — | Knowledge distillation from a teacher model |
| `--progressive` | — | Progressive multi-tier training (curriculum) |
| `--lora` | — | LoRA adapter training (forces `use_lora=true`, overrides config) |
| `--config` | `config/base.yaml` | Config file path |
| `--distributed` | off | Multi-GPU distributed training |
| `--num-epochs` | from config | Override training epochs |
| `--batch-size` | from config | Override per-GPU batch size |
| `--lr` | from config | Override learning rate |
| `--experiment-name` | auto | Name for this training run |
| `--checkpoint` | none | Resume from checkpoint path |
| `--force` | off | Ignore training checkpoint, re-train from scratch |
| `--start-tier` | none | Progressive: start from specific tier (`tier1`–`tier4`) |
| `--validate-final` | off | Progressive: validate final model |
| `--teacher` | `facebook/nllb-200-3.3B` | Teacher model for distillation |
| `--distill-alpha` | `0.5` | CE vs KD loss weight (0–1) |
| `--distill-temp` | `4.0` | Distillation temperature |

## Config Reference

Key settings in `config/base.yaml`:

```yaml
model:
  max_vocab_size: 32000
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  decoder_dim: 512
  decoder_layers: 8
  decoder_heads: 8

training:
  use_lora: false
  num_epochs: 10
  lr: 3e-4
  warmup_steps: 1000
  weight_decay: 0.01
  batch_size: 32
  accumulation_steps: 4
  effective_batch_size: 128
  gradient_checkpointing: true
  mixed_precision: true
  dtype: bfloat16
  compile_model: true
```

## Memory Usage (A100 40GB)

| Config | VRAM | Batch | Speed |
|---|---|---|---|
| Full model, no optimizations | ~18 GB | 32 | baseline |
| + gradient checkpointing | ~12 GB | 32 | ~1.2× slower |
| + mixed precision (bf16) | ~8 GB | 32 | ~1.5× faster |
| + compile | ~10 GB | 32 | ~2× faster |
| **All optimizations** | **~10 GB** | **32** | **~2-3× faster** |

## Monitoring

```bash
# Watch training progress
tail -f logs/training/latest.log

# Watch GPU usage
watch -n1 nvidia-smi

# Compare experiments
./uts tools --validate-config config/base.yaml
```

## Checkpoint Structure

```
checkpoints/{experiment_name}/
├── best_model.pt           # Best validation loss
├── checkpoint_epoch_*.pt   # Per-epoch snapshots
├── training_state.json     # Optimizer state, epoch, step
└── config.yaml             # Config used for this run
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Loss stays high (~11) | Model not learning | Check `use_lora: false` in config |
| Loss = NaN | LR too high | Reduce `lr` to `1e-4` |
| CUDA OOM | Batch too large | Reduce `batch_size` or increase `accumulation_steps` |
| Slow training | No compilation | Set `compile_model: true` |
| Validation loss > training | Overfitting | Increase data or reduce epochs |
