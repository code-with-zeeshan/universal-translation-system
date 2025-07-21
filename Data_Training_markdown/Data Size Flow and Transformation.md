# Data Size Flow and Transformation

This document describes how data flows through the Universal Translation System pipeline, from raw downloads to final training-ready datasets.

## 1. Data Sources
- Parallel corpora (OPUS, Tatoeba, FLORES-200, etc.)
- Downloaded using `data/download_training_data.py` or as part of the integrated pipeline

## 2. Configuration
- All supported languages and training pairs are defined in `data/config.yaml`
- Data size targets and quality thresholds are set in the same config

## 3. Pipeline Steps

### a. Download
- Raw data is downloaded to `data/raw/`
- Both curated and bulk data are supported

### b. Preprocessing
- Data is cleaned, filtered, and normalized using `data/practical_data_pipeline.py`
- Length filtering, deduplication, and quality checks are applied

### c. Sampling & Augmentation
- Smart sampling (`smart_sampler.py`) selects high-quality pairs
- Synthetic augmentation (`synthetic_augmentation.py`) adds backtranslations and pivot data

### d. Vocabulary Creation
- Vocabulary packs are generated from processed data using `vocabulary/create_vocabulary_packs_from_data.py`
- Packs are grouped by script or language family as defined in `data/config.yaml`

### e. Final Assembly
- Final training files are created in `data/processed/` and `data/final/`
- Data is validated for size and quality before training

## 4. Monitoring & Validation
- Data pipeline logs and stats are available in `logs/`
- Use the coordinator dashboard and Prometheus metrics to monitor data flow and system health

---

For a step-by-step execution flow, see [step-by-step Execution flow.md](step-by-step%20Execution%20flow.md).