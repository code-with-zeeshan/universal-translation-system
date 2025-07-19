## Data Size Flow and Transformation

The pipeline transforms and filters data at each stage. Here's how the data flows and transforms:

### 📊 Data Size Transformation Table

| Stage | Size | Status | What Happens | Kept/Discarded |
|-------|------|--------|--------------|----------------|
| **1. Essential Data** | ~100MB | ✅ **Kept Separate** | Evaluation sets | **Kept for testing** |
| **2. Raw Training Data** | ~50GB | ❌ **Temporary** | Downloaded raw data | **Discarded after sampling** |
| **3. Sampled Data** | ~5GB | ✅ **Kept** | High-quality filtered | **Part of final 8GB** |
| **4. Augmented Data** | ~3GB | ✅ **Kept** | Synthetic additions | **Part of final 8GB** |
| **5. Final Training Data** | **~8GB** | ✅ **Final Output** | Sampled + Augmented | **This is what you train on** |

### 🔄 Data Pipeline Transformation Flow

```
Step 1: Download Raw Data (~50GB)
         ↓ [Quality Filtering & Sampling]
Step 2: High-Quality Samples (~5GB) ← Discard ~45GB of low quality
         ↓ [Keep]
Step 3: Generate Synthetic Data (~3GB)
         ↓ [Merge]
Step 4: Final Training Corpus (~8GB)

Evaluation Data (~100MB) → Kept separately for testing
```

### 📁 Final Directory Structure and Sizes

| Directory | Contents | Size | Fate |
|-----------|----------|------|------|
| `data/essential/` | FLORES-200, Tatoeba samples | ~100MB | ✅ **Kept** (for evaluation) |
| `data/raw/` | Original downloads | ~50GB | ❌ **Can be deleted** |
| `data/sampled/` | Quality-filtered pairs | ~5GB | ⚠️ **Intermediate** (can delete after final) |
| `data/final/` | Augmented + pivot pairs | ~3GB | ⚠️ **Intermediate** (can delete after final) |
| **`data/processed/`** | **Final training corpus** | **~8GB** | ✅ **KEEP - This is your dataset** |

### 💾 Storage Requirements During Processing

| Phase | Storage Needed | Explanation |
|-------|----------------|-------------|
| During download | ~60GB | Raw data (50GB) + workspace |
| During sampling | ~55GB | Raw (50GB) + sampled (5GB) |
| During augmentation | ~13GB | Sampled (5GB) + augmented (3GB) + workspace |
| **Final (minimum)** | **~8.1GB** | Training (8GB) + evaluation (0.1GB) |
| **Final (recommended)** | **~13GB** | Keep sampled/augmented for debugging |

### 🎯 What You Actually Get

```yaml
Final Dataset (8GB):
├── High-quality sampled pairs: ~5GB
│   ├── en-es: 2M sentences (filtered from ~10M)
│   ├── en-fr: 2M sentences (filtered from ~10M)
│   └── ... other pairs
│
└── Synthetic augmented pairs: ~3GB
    ├── Backtranslated pairs
    └── Pivot-generated pairs

Evaluation Set (100MB) - Separate:
├── FLORES-200: High-quality test/dev sets
└── Tatoeba: Additional evaluation pairs
```

### 🗑️ Space Optimization Timeline

```bash
# After downloads complete
rm -rf data/raw/  # Free ~50GB

# After final corpus generation
rm -rf data/sampled/  # Free ~5GB (optional)
rm -rf data/final/    # Free ~3GB (optional)

# Minimum space after cleanup: ~8.1GB
```

**Summary**: The 50GB raw data is filtered down to 5GB of high-quality pairs, then augmented with 3GB of synthetic data, resulting in a final 8GB training corpus. The large raw downloads can be deleted after processing.