# Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Edge Device                         │
│  ┌───────────────────────────────────────────────┐  │
│  │  Universal Encoder (42.7M params)             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │  │
│  │  │ Embed   │  │ RoPE    │  │ 6× Transformer│  │  │
│  │  │ (32K)   │─>│ +       │─>│ Encoder       │─>│  │
│  │  │         │  │ SwiGLU  │  │ Layers (8H)   │  │  │
│  │  └─────────┘  └─────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                          │                           │
│                    encoder_hidden                     │
│                   (<1 KB / req)                      │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                  Cloud Decoder                       │
│  ┌───────────────────────────────────────────────┐  │
│  │  Optimized Universal Decoder (108.1M params)  │  │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │  │
│  │  │ Embed   │  │ Adapter │  │ 8× Decoder   │  │  │
│  │  │ (32K)   │─>│ (encoder│─>│ Layers (12H) │─>│  │
│  │  │         │  │ →decoder│  │ + cross-attn │  │  │
│  │  └─────────┘  └─────────┘  └──────────────┘  │  │
│  │                                                │  │
│  │  Target Language Adapters (one per language)   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Model Specs

| Property | Encoder | Decoder |
|---|---|---|---|
| Parameters | 42.7M | 108.1M |
| Hidden dim | 512 | 512 (unified) |
| Layers | 6 | 8 |
| Attention heads | 8 | 8 |
| FFN | 2048 (SwiGLU) | 2048 (ReLU) |
| Positional encoding | RoPE | Learned |
| Normalization | Pre-LN | Pre-LN |
| Vocab size | 32K (dynamic) | 32K (dynamic) |
| Inference | On-device (CPU/GPU) | Cloud (GPU) |
| Language ID | 20-lang embedding added per-token | — |

## Vocabulary System

6 per-script SentencePiece BPE packs, 32K tokens each:

| Pack | Languages | Size |
|---|---|---|
| `latin` | en, es, fr, de, it, pt, nl, sv, pl, id, vi, tr | ~32K |
| `cjk` | zh, ja, ko | ~32K |
| `arabic` | ar | ~32K |
| `devanagari` | hi | ~32K |
| `cyrillic` | ru, uk | ~32K |
| `thai` | th | ~32K |

Only needed packs are downloaded to the edge device. The embedding table dynamically resizes to match the loaded pack (up to `max_vocab_size: 32000`).

## Training Design

### Phase 1: Full Backbone (current)
- All 150.8M parameters train
- 20 languages jointly
- ~6 hours on A100 (10 epochs default), ~3 hours (5 epochs for quick testing)
- Target: BLEU 15-25 per pair

### Phase 2: LoRA Adapters (future languages)
- Backbone frozen
- LoRA adapters (r=16 encoder, r=64 decoder)
- Target language adapters (512→64→512 per language)
- ~2-3 hours on L4

## Data Flow

```
Raw datasets (opus-100, CCMatrix, ParaCrawl, etc.)
    │
    ▼ download_training
data/raw/{pair}.txt
    │
    ▼ sample_filter
data/processed/sampled/{pair}_sampled.txt  (deduplicated, quality-scored)
    │
    ├─▶ augment (optional)
    │      False friends, idioms, equivalence pairs, backtranslation, pivots
    │      Wikipedia mono-lingual data downloaded as backtranslation source
    │      NLLB-1.3B model (shared with distillation)
    │
    ▼ create_ready
datasets/train_final.txt  ──┐
datasets/val_final.txt    ──┤  (90/10 seed-shuffled split)
                           │
    ▼ vocabulary            ▼ training
vocabulary/vocab/          encoder + decoder
{script}_v*.msgpack        teacher forcing → best_model.pt
                           ↓
                           Evaluation (BLEU/COMET per language pair)

All data sources (sampled + augment + pivot + distilled) are merged by
PipelineConnector.create_final_training_file() into datasets/train_final.txt
and datasets/val_final.txt. The corpus/ files for vocabulary training are
extracted from the same merged sources, ensuring vocabulary coverage matches
the training data distribution.
```

## Serving Architecture

```
Client → Coordinator (load balancer)
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
  Decoder  Decoder  Decoder  (pool, auto-scaled)
  Node 1   Node 2   Node 3
    
Redis: pool registry, rate limiting, health state
```

The coordinator routes encoder hidden states to the least-loaded decoder node. Decoder nodes register via Redis and report health metrics. Prometheus/Grafana monitor all endpoints.
