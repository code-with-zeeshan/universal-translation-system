# Publishing Guide

Publish a trained model to Hugging Face Hub for distribution to SDKs and decoder nodes.

## Quick Start

```bash
# Auto-discover latest checkpoint and publish
uts publish --repo-id your-org/universal-translation-system

# Specify checkpoint explicitly
uts publish --repo-id your-org/uts --checkpoint checkpoints/exp1/best_model.pt
```

## Workflow Steps

When you run `uts publish`, the following happens automatically:

```
checkpoint.pt
    │
    ├─ 1. Split → encoder.pt + decoder.pt
    ├─ 2. Export → encoder.onnx (unless --no-onnx)
    ├─ 3. Quantize (unless --no-quantize)
    ├─ 4. Verify artifacts
    └─ 5. Upload to HF Hub
```

### 1. Find & Split Checkpoint

The publish script finds `best_model.pt` under `checkpoints/` (latest by mtime). It splits the state dict into separate `encoder.pt` and `decoder.pt` files saved to `models/production/`.

### 2. ONNX Export

The encoder is exported to ONNX format (opset 17) with dynamic batch/sequence axes. This enables:
- ONNX Runtime inference on CPU
- WebAssembly deployment via ONNX Runtime Web
- Mobile inference via CoreML/ONNX conversion

### 3. Quantization

Runs the quantization pipeline (`training.quantization_pipeline`) to produce INT8 quantized variants for smaller footprint.

### 4. Upload to HF Hub

Uploads all artifacts via `scripts/upload_artifacts.py`. Repository layout after upload:

```
your-org/universal-translation-system/
├── models/
│   └── production/
│       ├── encoder.pt
│       ├── decoder.pt
│       └── encoder.onnx
├── vocabs/
│   └── *.msgpack          # 6 per-script vocabulary packs
└── adapters/
    └── *.pt               # Target language adapters
```

## CLI Reference

```bash
uts publish --repo-id <HUB_REPO_ID> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--repo-id` | (required) | HF Hub repo ID (e.g., `your-org/universal-translation-system`) |
| `--checkpoint PATH` | auto-discover | Path to `best_model.pt` |
| `--no-onnx` | off | Skip ONNX export |
| `--no-quantize` | off | Skip quantization |
| `--upload-only` | off | Upload existing artifacts without reprocessing |
| `--preflight` | — | Run cloud preflight checks (separate action) |
| `--optimize-decoder` | — | Run decoder quantization/optimization (separate action) |

### Preflight Checks

Before publishing, validate that the deployment environment is ready:

```bash
uts publish --preflight
```

Checks cloud configuration, HF Hub authentication, and dependency availability.

### Optimize Decoder

Run additional decoder optimization (quantization, graph optimization):

```bash
uts publish --optimize-decoder
```

## HF Hub Authentication

Set the `HF_TOKEN` environment variable or log in via `huggingface-cli login`:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

The repository must already exist on HF Hub (create it at [huggingface.co/new](https://huggingface.co/new)).

## Upload-Only Mode

If you've already processed artifacts and just need to re-upload:

```bash
uts publish --repo-id your-org/uts --upload-only
```

This skips split/export/quantize and uploads whatever is in `models/production/`.

## Related Docs

- `docs/DEPLOYMENT.md` — Serving published models
- `docs/SDK_INTEGRATION.md` — Using published models in SDKs
- `scripts/upload_artifacts.py` — Upload implementation
