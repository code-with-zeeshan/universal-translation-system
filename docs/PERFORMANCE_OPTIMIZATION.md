# Performance Optimization

## Training Optimizations (A100 40GB)

### Recommended Settings for Full Training

```yaml
training:
  mixed_precision: true        # bfloat16 — 2× memory savings, no quality loss
  dtype: bfloat16
  gradient_checkpointing: true # Trade compute for memory (~40% savings)
  compile_model: true          # JIT compile decoder — 2-3× speedup
  flash_attention: true        # Linear memory attention (O(n) vs O(n²))
  batch_size: 32               # Fits in 40GB with all optimizations
  accumulation_steps: 4        # Effective batch = 128
```

### Speed Comparison

| Config | Time/epoch (A100) | VRAM | Batch |
|---|---|---|---|
| No optimizations | ~55 min | ~18 GB | 32 |
| + mixed precision | ~40 min | ~12 GB | 32 |
| + gradient checkpointing | ~50 min | ~8 GB | 32 |
| + compile | ~25 min | ~10 GB | 32 |
| **All optimizations** | **~30 min** | **~10 GB** | **32** |

### Inference Optimizations

```yaml
model:
  flash_attention: true   # Faster cross-attention in decoder
decoder:
  compile_model: true      # 2× faster generation
```

## Memory Profile (Full Training)

| Component | Size |
|---|---|
| Model weights (fp32) | ~600 MB |
| Optimizer states (Adam) | ~1.2 GB |
| Activations (checkpointed) | ~500 MB |
| Gradients | ~600 MB |
| **Total** | **~2.9 GB** (well within 40 GB) |

## GPU Selection

| GPU | Full Training (10 epochs) | LoRA Training (5 epochs) |
|---|---|---|
| A100 40GB | ~6 hours | ~1.5 hours |
| L4 24GB | ~10 hours | ~2.5 hours |
| L40s 48GB | ~4 hours | ~1 hour |
| T4 16GB | ~18 hours | ~4 hours |

## Pipeline Optimizations

- **Skip finished stages**: `--resume` checks each stage before running
- **Download only eval**: `--eval-only` skips training data
- **Incremental augment**: Add stages incrementally
- **Parallel downloads**: Datasets downloaded concurrently where possible

## Evaluation Optimizations

- **Batched inference**: Default batch 32, tuned for 2K eval samples
- **Length-sorted batches**: Minimizes padding waste
- **Single GPU**: Sufficient for eval (no distribution needed)
