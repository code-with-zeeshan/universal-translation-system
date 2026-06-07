# FAQ

## Training

### Why is my BLEU score near zero after 1 epoch?
The model needs **5-10 epochs** to converge. One epoch of LoRA on a random backbone gives ~0.001 BLEU. Either train longer or disable LoRA for full model training.

### Should I use LoRA?
- **For initial training (20 languages):** No. Set `use_lora: false` and train all 150.8M params.
- **For adding new languages:** Yes. Freeze backbone, train LoRA adapters only.

### What GPU do I need?
- **A100 40GB** (recommended) — full 150.8M model, batch 32, ~3 hours (5 epochs)
- **L4 24GB** — full model, smaller batch, ~10 hours
- **L40s 48GB** — fastest but expensive

### How much data is needed?
~2-5M sentence pairs across 20 languages from opus-100. Each language pair has 50K-200K training sentences.

## Data Pipeline

### The pipeline is stuck at "Downloading..."
Hugging Face dataset downloads take time. Check `data/raw/*.txt` to see if files are growing. Use `--resume` if interrupted.

### How do I add a new dataset?
Add to `data/unified_data_downloader.py` and register in config `data.sources`. See existing sources (opus-100, etc.) as templates.

## Evaluation

### Which metrics are reported?
SacreBLEU (tokenized) per language pair. COMET is scaffolded but not fully integrated.

### How do I evaluate a single language pair?
```bash
python -m evaluation.evaluate_model \
  --checkpoint checkpoints/*/best_model.pt \
  --test-data data/evaluation/en-es.tsv
```

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

## SDK & Deployment

### Are the SDKs ready for production?
Android, iOS, Flutter, React Native, and Web SDKs are scaffolded but need a trained encoder binary. See `sdk/` directory.

### How do I deploy the decoder?
```bash
./uts serve --setup --all
./uts serve --decoder
./uts serve --coordinator
```
