# Performance Optimization

## GPU Tier System

The pipeline auto-detects your GPU hardware and applies optimal settings for every stage.
Override with `UTS_GPU_TIER=t4|l4|l40s|a100|h100|cpu` or set explicit values in `config/base.yaml` (`0` = auto).

| Tier | Memory | GPUs | Typical Hardware |
|------|--------|------|-----------------|
| `t4` | 16 GB | 1 | Colab T4, GCP T4 |
| `l4` | 24 GB | 1 | GCP L4, AWS L4 |
| `l40s` | 48 GB | 1+ | NVIDIA L40S |
| `a100` | 80 GB | 1-8 | NVIDIA A100 (40/80 GB), A10 |
| `h100` | 80 GB | 1-8 | NVIDIA H100, H200 |
| `cpu` | — | 0 | CPU-only, no CUDA |

### Per-Tier Optimal Settings

| Setting | T4 | L4 | L40S | A100 | H100 |
|---------|-----|-----|------|------|------|
| **Download workers** | 8 | 12 | 24 | 32 | 48 |
| **OPUS rate limit** | 8/s | 10/s | 15/s | 20/s | 30/s |
| **NLLB batch size** | 128 | 256 | 512 | 1024 | 2048 |
| **NLLB dtype** | fp16 | fp16 | fp16 | bf16 | bf16 |
| **Flash Attention 2** | yes | yes | yes | yes | yes |
| **BetterTransformer** | yes | yes | yes | yes | yes |
| **torch.compile** | — | yes | yes | yes | yes |
| **COMET batch size** | 64 | 128 | 256 | 512 | 1024 |
| **LaBSE batch size** | 64 | 128 | 256 | 512 | 1024 |
| **Create-ready workers** | 4 | 8 | 12 | 16 | 24 |
| **Vocab SP threads** | 8 | 12 | 16 | 32 | 48 |

---

## Data Pipeline Optimizations

### 1. Download Training Data (CPU/IO-bound)

- **Max workers** auto-scaled per GPU tier, clamped to `os.cpu_count()`
- **Parallel batches** enabled by default (was sequential)
- **OPUS rate limit** scaled per tier (T4: 8/s, H100: 30/s vs old hardcoded 2/s)
- Per-worker HTTP sessions with retry + backoff on 429

### 2. Sample & Filter (CPU + GPU)

- Heuristic filters (length, ratio, numbers, quality) run in `ProcessPoolExecutor`
- **LaBSE embedding similarity filter** (GPU): cross-lingual cosine similarity via SentenceTransformer, batch size per tier
- Low-similarity pairs dropped (threshold 0.5), freeing disk space and improving training signal

### 3. Augment — NLLB Backtranslation (GPU-bound)

- **Flash Attention 2**: enabled on all tiers — 2-3x faster generation vs eager attention
- **BetterTransformer**: enabled on all tiers — kernel fusion for transformer inference
- **torch.compile**: enabled on L4+ (reduce-overhead mode) — 1.5-2x additional speedup
- **Batch probing**: progressive inference at increasing batch sizes until OOM, then steps back for 25% headroom
- **OOM recovery**: if generation OOMs at runtime, batch halves and retries (up to 3x)
- **Multi-GPU**: A100/H100 profiles set `nllb_multi_gpu=True` (tensor parallelism via `device_map="auto"`)

### 4. Knowledge Distillation (GPU-bound)

Same NLLB optimizations as augmentation (FA2, BetterTransformer, compile, tier-scaled batch).

### 5. COMET Quality Filter (GPU-bound)

- **Dynamic batch size**: per-tier (T4: 64, A100: 512, H100: 1024)
- **Parallel file processing**: train + val scored concurrently (was sequential)

### 6. Create Training-Ready (CPU/IO-bound)

- **Parallel I/O**: `create_monolingual_corpora()` + `create_final_training_file()` run concurrently via `ThreadPoolExecutor`
- Workers per tier (T4: 4, H100: 24)

### 7. Vocabulary (CPU-bound)

- SentencePiece `num_threads` auto-scaled per GPU tier (T4: 8, A100: 32, H100: 48)
- Configurable via `vocab_threads: 0` in `base.yaml` (0 = auto)

### 8. Wikipedia Download

- 20 languages downloaded concurrently (ThreadPoolExecutor, `WIKI_DOWNLOAD_WORKERS=20`)
- Streaming from HuggingFace datasets (no disk for intermediate files)

---

## Est. Speedup vs Old Defaults (Per Tier)

| Pipeline Segment | T4 | L4 | L40S | A100 | H100 |
|-----------------|-----|-----|------|------|------|
| Download | 1.5-2x | 2-3x | 3-4x | 4-6x | 6-8x |
| Sample filter | 1.5x | 2x | 4x | 8x | 16x |
| NLLB augment | 2-3x | 3-4x | 4-6x | 6-10x | 8-16x |
| COMET filter | 1x | 2x | 4x | 8x | 16x |
| Create ready | 1.5x | 2x | 3x | 4x | 6x |
| Vocabulary | 1x | 1.5x | 2x | 4x | 6x |
| **Total pipeline** | **~2-2.5x** | **~3-4x** | **~4-6x** | **~6-10x** | **~8-15x** |

---

## Training Optimizations

### Recommended Settings

```yaml
training:
  mixed_precision: true        # bfloat16 — 2x memory savings, no quality loss
  dtype: bfloat16
  gradient_checkpointing: true # 40% memory savings at compute cost
  compile_model: true          # JIT compile decoder — 2-3x speedup
  flash_attention: true        # O(n) vs O(n²) memory for attention
  batch_size: 32               # Fits in 40GB with all optimizations
  accumulation_steps: 4        # Effective batch = 128
```

### Speed Comparison (A100 40GB)

| Config | Time/epoch | VRAM | Batch |
|--------|-----------|------|-------|
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
  compile_model: true      # 2x faster generation
```

## Memory Profile (Full Training)

| Component | Size |
|-----------|------|
| Model weights (fp32) | ~600 MB |
| Optimizer states (Adam) | ~1.2 GB |
| Activations (checkpointed) | ~500 MB |
| Gradients | ~600 MB |
| **Total** | **~2.9 GB** (well within 40 GB) |

## GPU Selection Guide

| GPU | Full Training (10 epochs) | LoRA Training (5 epochs) | Data Pipeline |
|-----|--------------------------|--------------------------|---------------|
| A100 80GB | ~6 hours | ~1.5 hours | ~1-2 hours |
| L40S 48GB | ~4 hours | ~1 hour | ~2-3 hours |
| L4 24GB | ~10 hours | ~2.5 hours | ~3-5 hours |
| T4 16GB | ~18 hours | ~4 hours | ~4-6 hours |

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UTS_GPU_TIER` | auto | Force GPU profile: `t4`/`l4`/`l40s`/`a100`/`h100`/`cpu` |
| `WIKI_DOWNLOAD_WORKERS` | `20` | Concurrent Wikipedia downloads |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Wrong GPU profile detected | New/unknown GPU | Set `UTS_GPU_TIER` explicitly |
| NLLB OOM even at batch_size=16 | Insufficient GPU memory | Switch to NLLB-600M: set `base_model: facebook/nllb-200-distilled-600M` in config |
| BetterTransformer not applied | `optimum` not installed | `pip install optimum` |
| torch.compile fails | PyTorch < 2.0 or incompatible GPU | Falls back silently (check logs for "torch.compile failed") |
| Flash Attention 2 not applied | `flash-attn` not installed or GPU not supported | Falls back to eager attention (check logs) |
| LaBSE embedding filter skipped | `sentence-transformers` not installed | Falls back to heuristic-only filtering |
