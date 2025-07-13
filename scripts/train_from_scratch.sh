#!/bin/bash
# scripts/train_from_scratch.sh

echo "🚀 Starting Universal Translation System Training"

# 1. Download and prepare data
echo "📥 Step 1: Downloading training data..."
python data/download_training_data.py

# 2. Preprocess data
echo "🔧 Step 2: Preprocessing data..."
python data/preprocess_parallel_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --min_length 5 \
    --max_length 200

# 3. Create vocabulary packs
echo "📦 Step 3: Creating vocabulary packs..."
python vocabulary/create_vocabulary_packs_from_data.py

# 4. Initialize models from pretrained
echo "🧠 Step 4: Initializing models from pretrained..."
python training/bootstrap_from_pretrained.py

# 5. Train models
echo "🏃 Step 5: Training models..."
python training/train_universal_system.py \
    --encoder_path models/universal_encoder_initial.pt \
    --decoder_path models/universal_decoder_initial.pt \
    --train_data data/processed/train.txt \
    --val_data data/processed/val.txt \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5

# 6. Convert models for production
echo "🔄 Step 6: Converting models..."
python training/convert_models.py \
    --encoder_checkpoint checkpoints/best_model.pt \
    --decoder_checkpoint checkpoints/best_model.pt \
    --output_dir models/production

# 7. Optimize for mobile
echo "📱 Step 7: Optimizing for mobile..."
python training/optimize_for_mobile.py \
    --onnx_model models/production/encoder.onnx \
    --output_dir models/mobile

echo "✅ Training complete!"