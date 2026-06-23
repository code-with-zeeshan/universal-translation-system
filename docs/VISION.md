# Universal Translation System: Our Vision

## The Core Innovation: Symmetric Split Architecture

Our Universal Translation System is built on a split architecture that solves multiple problems elegantly:

```
DEVICE                                   CLOUD
+----------------------------+         +----------------------+
| Universal Encoder           |         | Universal Decoder    |
| • 42.7M params              |         | • 108.1M params      |
| • 512-dim, 6 layers, 8 heads| =====> | • 768-dim, 8 layers, |
| • 171MB FP32 / 86MB FP16   |  <1KB   |   12 heads           |
|   / 43MB INT8              |embedding | • All languages      |
| • ONNX + NNAPI / CoreML    |         |   served by one model|
| + Vocabulary Packs:         |         |                      |
|   • 32K BPE per script     |         |                      |
|   • 2–4MB each             |         |                      |
|   • 6 packs: latin, cjk,   |         |                      |
|     cyrillic, arabic,      |         |                      |
|     devanagari, thai       |         |                      |
+----------------------------+         +----------------------+
```

## Key Design Decisions

### One Model, Multiple Deployments
- Train a single 42.7M encoder (512 dims, 6 layers, RoPE + SwiGLU)
- Deploy at different precision levels:
  - **Cloud**: Float32 (171MB) - 100% quality
  - **Flagship phones**: Float16 (86MB) - 99% quality  
  - **Standard phones**: INT8 (43MB) - 97% quality
- Users choose based on their device/preference

### Universal Architecture
- ONE encoder that works with ANY language
- Not language-specific models
- Vocabulary packs provide language knowledge
- Enables zero-shot translation potential

### Symmetric Embedding Space
- Encoder: 512 dimensions
- Decoder: 768 dimensions (cross-attends to encoder hidden states via 512→768 projection)
- Per-language adapter bottlenecks (768→96→768) after each decoder layer
- Gradient checkpointing enabled by default for memory efficiency

## The User Experience

1. **Download app**: Base encoder (43MB or 86MB based on device)
2. **Select languages**: Download only needed vocabulary packs (2-4MB each)
3. **Type text**: Encoder creates embeddings using vocabulary pack
4. **Privacy-preserving processing**: 
   - Embeddings compressed to &lt;1KB
   - Sent to cloud (not raw text!)
   - Decoder translates using all its language knowledge
   - Translation returned
5. **Total size**: ~50MB for standard phone with one language pack

## Why This Architecture is Revolutionary

### Traditional Approaches:
- Full models per language pair: 100MB x 20 languages = 2GB
- Cloud-only: Privacy concerns, needs constant internet
- Edge-only: Huge apps, limited device capability

### Our Approach:
- Shared universal encoder: 43-171MB total
- Dynamic vocabulary loading: 2-4MB per language group
- Privacy-preserving: Only embeddings leave device
- Optimal compute split: Light encoding edge, heavy decoding cloud
- Zero-retraining expansion: LoRA adapters (~7M params) for new languages

## Current Implementation

This vision has been fully implemented with a **512-dim** encoder and **768-dim** decoder:

| Component | Actual | Status |
|---|---|---|
| Encoder | 42.7M params, 512-dim, 6 layers, RoPE + SwiGLU | ✅ `runtime/encoder/` |
| Decoder | 108.1M params, 768-dim, 8 layers, cross-attention | ✅ `runtime/cloud_decoder/` |
| Coordinator | Load balancer, circuit breakers, 50ms batching, A/B testing | ✅ `runtime/coordinator/` |
| Vocab Packs | 6 packs (latin, cjk, cyrillic, arabic, devanagari, thai) | ✅ `runtime/vocabulary/` |
| C++ Edge Encoder | ONNX Runtime with NNAPI / CoreML / XNNPACK delegates | ✅ `runtime/encoder_core/` |
| Data Pipeline | Auto-resume, config-hash checkpointing, 6 stages | ✅ `pipeline/data/` |
| Training | Full, distillation (NLLB-1.3B teacher), progressive, LoRA | ✅ `pipeline/training/` |
| SDKs | Android, iOS, Flutter, React Native, Web | ✅ `sdk/` |
| Deployment | Docker Compose, Kubernetes, Helm, Prometheus/Grafana | ✅ `deploy/` |
| TUI Dashboard | Real-time pipeline + training + GPU monitoring | ✅ `tui/` |

For technical details, see `docs/ARCHITECTURE.md` and `docs/environment-variables.md`.
