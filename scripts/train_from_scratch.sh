#!/bin/bash
# scripts/train_from_scratch.sh

set -e

echo "🚀 Starting Universal Translation System Training"

# 1. Prepare all data (download, preprocess, sample, augment, etc.)
echo "📥 Step 1: Running integrated data pipeline..."
python data/practical_data_pipeline.py

# 2. Create vocabulary packs
echo "📦 Step 2: Creating vocabulary packs..."
python vocabulary/create_vocabulary_packs_from_data.py

# 3. Initialize models from pretrained
echo "🧠 Step 3: Initializing models from pretrained..."
python training/bootstrap_from_pretrained.py

# 4. Train models (config auto-detection is now built-in)
echo "🏃 Step 4: Training models..."
python training/train_universal_system.py

# 5. Convert models for production
echo "🔄 Step 5: Converting models..."
python training/convert_models.py \
    --encoder_checkpoint checkpoints/best_model.pt \
    --decoder_checkpoint checkpoints/best_model.pt \
    --output_dir models/production

# 6. Optimize for mobile
echo "📱 Step 6: Optimizing for mobile..."
python training/optimize_for_mobile.py \
    --onnx_model models/production/encoder.onnx \
    --output_dir models/mobile

echo "✅ Training complete!"