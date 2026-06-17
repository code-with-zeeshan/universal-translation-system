# AGENTS.md — Universal Translation System

## Project Overview

A panacea that unifies all core translation system components into one pipeline: data ingestion, cleaning, augmentation, vocabulary management, model training (fine-tuning / LoRA on NLLB), and evaluation. Work is driven by `pipeline/main.py` which orchestrates stages via `orchestrator.py`.

## System Architecture

```
pipeline/main.py  (entry point)
├── cli.py            — CLI argument parsing (--stage, --gpu-level, etc.)
├── data/
│   ├── orchestrator.py — state machine, checkpoint/resume, dispatch
│   ├── ingestion.py    — download_training, download_wikipedia (HuggingFace datasets)
│   ├── cleaning.py     — dedup, language ID, quality filtering (sample_filter)
│   ├── augmentation.py — backtranslation (NLLB), false friends, idioms, tone/register
│   └── domain_datasets.py — domain data generation for cultural context
├── model/
│   └── train.py        — LoRA fine-tuning of NLLB-600M (main target)
├── vocabulary/
│   └── evolve.py       — vocabulary evolution / augmentation (DEPRECATED? used by augment stage via `--vocab`)
└── evaluation/
    └── evaluate.py     — BLEU, COMET, etc.
```

## Pipeline Stages (in order)

1. **download_training** — Downloads opus-100 (14 primary pairs) + Wikipedia + additional OPUS pairs
2. **download_wikipedia** — Downloads Wikipedia monolingual data for each language
3. **sample_filter** — Sampling + quality filtering (laBSE embeddings, LaBSE + LASER fallback)
4. **augment** — Backtranslation (NLLB), false friends, idioms, tone/register
5. **tokenize** — Tokenization into SPM subword units
6. **train** — LoRA fine-tuning of NLLB-600M
7. **evaluate** — BLEU / COMET

## Weight Initialization

- LoRA adapters are initialized from the saved LoRA checkpoint with scaling removed
- Base model is always locked to pretrained NLLB weights
- The model itself is never fine-tuned fully — only LoRA adapters are trained

## When to Load AGENTS.md

Always read this file at the start of every conversation to understand where we left off.

## Current Status (June 17, 2026)

### ✅ Done
- Scaffolded entire project structure (pipeline, tests, configs, scripts)
- Defined orchestrator state machine (9 stages, checkpoint/resume, retry logic)
- Implemented all 7 pipeline stages + orchestrator wiring + CLI
- `/workspace` gating — prompts user if not running in expected Colab `/workspace`
- Colab notebook (`translation_pipeline.ipynb`) — one-click drive mount + run
- Fixed `NameError: name 'torch' is not defined` in `pipeline/vocabulary/evolve.py` — added `import torch`
- Fixed `KeyError: 'base'` in `orchestrator.py._load_checkpoint` / `_save_checkpoint` — changed `self.dirs['base']` → `self.runtime_dirs.data_dir`
- Fixed Wikipedia config date in `backtranslation.py` — `WIKIPEDIA_DATE` changed from `"20231101"` to `"20220301"` to match HF datasets library
- Fixed NLLB OOM issues in `augmentation.py`:
  - Probe now uses realistic-length text and `max_length=512` to match real generation memory
  - Probe step-back increased from 1→2 (25% headroom instead of 10%)
  - Probe limit halved (`total_gb * 12` instead of `* 25`)
  - OOM recovery now destroys the full model+translator and calls `gc.collect()` to fully release CUDA memory
  - `_process_backtranslation_batch` now has OOM recovery (previously failed silently)

### 🧪 Colab Run Results (June 17)
**download_training**: ✅ Success (~58 min). All 14 primary opus-100 pairs + additional OPUS pairs downloaded.
**sample_filter**: ✅ Success (~4 min). All pairs quality-filtered with 22–65% retention.
**augment**: ❌ Partial (before fixes). NLLB-1.3B loaded on T4 (14.56 GiB). False friends + idioms generated for en_es, en_fr, en_de. Backtranslation OOM — probe passed at bs=384 but actual generation OOM'd at 384→192→96→48 because the probe used short text (7 tokens) and small max_length (128).

### 🐞 Known Issues

1. ~~**Wikipedia download broken** — HuggingFace datasets library updated config versions (`20231101.*` → `20220301.*`). All 14 languages fail with "BuilderConfig '20231101.*' was not found"~~ ✅ **FIXED** in `backtranslation.py:28`

2. ~~**NLLB OOM on T4 16GB** — Batch probe underestimates generation memory. Backtranslation with NLLB-1.3B OOMs on 16GB T4 even at batch_size=48.~~ ✅ **FIXED** — probe now uses realistic text, max_length=512, 25% headroom, and OOM recovery destroys/reloads model.

3. **`_wizard_shared.py` references 3.3B model** — Still references `facebook/nllb-200-3.3B` in prompts (probably stale from earlier design)

4. **NLLB-1.3B may still OOM on T4** — Even with the improved probe, NLLB-1.3B's generation memory requirements may exceed 16GB for longer sequences. If backtranslation still fails, switch the augmenter's `base_model` to `facebook/nllb-200-distilled-600M`.

### 🔜 Next Steps
- Push to GitHub
- Run full pipeline `--stages download_training,sample_filter,augment,tokenize,train,evaluate`
- If augment still OOMs on T4, switch to NLLB-600M for the augment stage
