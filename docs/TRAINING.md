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
| `num_epochs` | 5 (default) | Budget-friendly; use `--num-epochs 10` for full convergence |
| `lr` | `3e-4` | Standard for full model training |
| `warmup_steps` | 1000 | Quick warmup to target LR |
| `batch_size` | 32 | Per-GPU, fits A100 40GB |
| `accumulation_steps` | 4 | Effective batch = 128 |

**Expected loss trajectory (default 5 epochs):**
- Epoch 0: ~11.0 (random init)
- Epoch 1: ~8.5
- Epoch 3: ~5.5
- Epoch 5: ~4.0 (default stop; `--num-epochs 10` continues to ~3.5)
- Epoch 10: ~3.0-3.5

**Expected BLEU after 10 epochs:** 15-25 (varies by language pair)

### Phase 2: LoRA Adapter Training (future languages)

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
| `--full` | — | Full model training (all params) |
| `--config` | `config/base.yaml` | Config file path |
| `--distributed` | off | Multi-GPU distributed training |
| `--num-epochs` | from config | Override training epochs |
| `--batch-size` | from config | Override per-GPU batch size |
| `--lr` | from config | Override learning rate |
| `--experiment-name` | auto | Name for this training run |
| `--checkpoint` | none | Resume from checkpoint path |

## Config Reference

Key settings in `config/base.yaml`:

```yaml
model:
  max_vocab_size: 32000
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  decoder_dim: 768
  decoder_layers: 8
  decoder_heads: 12

training:
  use_lora: false
  num_epochs: 5               # Default; --num-epochs 10 for longer
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
