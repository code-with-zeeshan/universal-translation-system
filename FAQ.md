# FAQ

## Training

### Why is my BLEU score near zero after 1 epoch?
The model needs **5-10 epochs** to converge. One epoch of LoRA on a random backbone gives ~0.001 BLEU. Either train longer or disable LoRA for full model training.

### Should I use LoRA?
- **For initial training (20 languages):** No. Set `use_lora: false` and train all 150.8M params.
- **For adding new languages:** Yes. Freeze backbone, train LoRA adapters only.

### What GPU do I need?
- **A100 40GB** (recommended) — full 150.8M model, batch 32, ~6 hours (10 epochs, $9.30)
- **L4 24GB** — full model, smaller batch, ~10 hours
- **L40s 48GB** — fastest but expensive
- **T4 16GB** — full model possible with batch 8, ~18 hours

### How much data is needed?
~2-5M sentence pairs across 20 languages from opus-100. Each language pair has 50K-200K training sentences. Use `--scale 5` for 5× the default data (~$42 total).

### What is knowledge distillation?
Distillation (`uts train --distill`) transfers knowledge from NLLB-200-3.3B teacher to the student via KL-divergence loss:
`loss = alpha * CE(student, labels) + (1-alpha) * T^2 * KL(student/T || teacher/T)`. Default alpha=0.5, temp=4.0.

### What is auto-resume?
The pipeline and trainer save checkpoints with config hashes. If interrupted, `uts data --pipeline` and `uts train --full` resume from the last completed step. Use `--force` to re-run from scratch. Config changes (epochs, LR, batch size) auto-invalidate downstream stages.

## Data Pipeline

### The pipeline is stuck at "Downloading..."
Hugging Face dataset downloads take time. Check `data/raw/*.txt` to see if files are growing. Auto-resume is the default — just re-run `uts data --pipeline` if interrupted.

### How do I add a new dataset?
Add to `pipeline/data/orchestrator.py` and register in config `data.sources`. See existing sources (opus-100, etc.) as templates.

### What does `--scale` do?
`--scale 5` multiplies training data targets by 5× (generates a temp config). For ~$42 total vs $11.25 default.

## Evaluation

### Which metrics are reported?
SacreBLEU (tokenized) per language pair. COMET and chrF are also computed when available.

### How do I evaluate a single language pair?
```bash
./uts eval --model --checkpoint checkpoints/*/best_model.pt
# Or directly:
python -m evaluation.evaluate_model \
  --checkpoint checkpoints/*/best_model.pt \
  --test-data data/evaluation/en-es.tsv
```

### Does eval auto-resume?
Yes. Evaluation tracks per-file completion. If interrupted, re-run with `uts eval --model` and it skips already-evaluated files. Use `--force` to re-evaluate all files.

## Architecture

### Can I run the encoder on CPU?
Yes. The 42.7M encoder is designed for edge devices. It runs on CPU at ~50 tokens/second.

### Can I run the decoder on CPU?
The 108.1M decoder works on CPU but is slow (~5 tokens/second). GPU recommended.

### How do I add a new language?
1. Add language code to `active_languages` in config
2. Assign to a script pack in `language_to_pack_mapping`
3. If new script, create a new pack group in `VocabularyConfig`
4. Train with `use_lora: true` for adapter-only training

### How does the coordinator work?
Routes decode requests to the least-loaded healthy decoder. When only one decoder is registered, SDKs call it directly (bypassing the coordinator). With multiple decoders, requests proxy through a 50ms batcher window for efficiency.

## CLI & Tools

### How do I see what files the system creates?
`uts docs --open layout` shows every directory and file created during operation.

### How do I check my environment?
`uts setup --check` runs GPU/environment readiness. `uts setup --verify` validates post-deployment setup.

### How do I check component versions?
`uts tools --version` shows all component versions from `version-config.json`. `uts tools --check-compat` runs API/schema/version compatibility checks.

### How do I rotate secrets?
`uts tools --rotate-secrets` rotates all 6 key secrets. Supports HS256 and RS256. Default rotation period: 90 days.

## SDK & Deployment

### Are the SDKs ready for production?
All 5 SDKs (Android, iOS, Flutter, React Native, Web) are enhanced with coordinator-aware routing, local decoder preference, and port auto-scan. They need a trained encoder binary for the edge encoding pipeline.

### How do I deploy the decoder?
```bash
# Quick start
uts serve --decoder

# Full stack
uts serve --setup --all
uts serve --decoder
uts serve --coordinator

# Or via UDN
udn start --host 0.0.0.0 --port 8001
```

### How do I publish a trained model?
```bash
uts publish --repo-id your-org/universal-translation-system
# Splits checkpoint → ONNX → quantize → upload to HF Hub
```

### What is the TUI dashboard?
`uts tui --config config/base.yaml` opens a real-time terminal UI showing pipeline stage progress, training metrics (loss, BLEU, LR, tokens/sec), GPU utilization, and a scrolling log. Keyboard shortcuts: `q` quit, `r` refresh, `h` help.
