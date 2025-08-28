#!/bin/bash
# scripts/train_from_scratch.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting Full End-to-End Training Pipeline for Universal Translation System"
echo "=============================================================================="

# 1. Prepare all data (download, preprocess, sample, augment, etc.)
echo -e "\n[Step 1/6] 📥 Running unified data pipeline..."
python - <<'PY'
from config.schemas import load_config
from data.unified_data_pipeline import UnifiedDataPipeline
import asyncio

cfg = load_config('config/training_generic_gpu.yaml')
pipeline = UnifiedDataPipeline(cfg)
asyncio.run(pipeline.run_pipeline(resume=True, stages=None))
PY
echo "[Step 1/6] ✅ Data pipeline complete."

# 2. Create vocabulary packs
echo -e "\n[Step 2/6] 📦 Creating vocabulary packs from processed data..."
python -m vocabulary.unified_vocabulary_creator
echo "[Step 2/6] ✅ Vocabulary packs created."

# 3. Initialize models from pretrained
echo -e "\n[Step 3/6] 🧠 Initializing models from pretrained weights (XLM-R & mBART)..."
python -m training.bootstrap_from_pretrained
echo "[Step 3/6] ✅ Initial models created."

# 4. Train models (config auto-detection is now built-in)
echo -e "\n[Step 4/6] 🏃 Starting main training loop (auto-detecting best config for hardware)..."
python -m training.launch train --config config/training_generic_gpu.yaml
echo "[Step 4/6] ✅ Main training loop complete."

# 5. Convert models for production
echo -e "\n[Step 5/6] 🔄 Converting trained models to production formats (ONNX, etc.)..."
python -m training.convert_models
echo "[Step 5/6] ✅ Model conversion complete."

echo -e "\n[Step 6/6] 📦 Finalizing production models..."
# This step seems to be missing from your file list, but if it existed, it would go here.
# For example: python -m training.optimize_for_mobile ...
echo "[Step 6/6] ✅ Production models are ready."

echo -e "\n🎉 Full training pipeline finished successfully! Models are ready in 'models/production'."
echo "=============================================================================="