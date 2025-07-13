# Training Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space
- 16GB+ RAM

## Data Preparation

### 1. Download Training Data

```bash
# Download minimal dataset for testing
python data/download_sample_data.py --languages "en,es,fr"

# Download full dataset (requires more space)
python data/smart_data_downloader.py --languages "all"
```

### 2. Create Vocabulary Packs
```bash
python vocabulary/create_vocabulary_packs_from_data.py
```

**This creates vocabulary packs for each language group**:

- **latin_v1.msgpack** - Latin script languages
- **cjk_v1.msgpack** - Chinese, Japanese, Korean
- etc.

## Model Training

**Option 1: From Scratch (Not Recommended)**
```bash
python training/train_from_scratch.py \
    --encoder_config configs/encoder_small.yaml \
    --decoder_config configs/decoder_small.yaml \
    --data_dir data/processed \
    --output_dir models/from_scratch
```

**Option 2: From Pretrained (Recommended)**
```bash
# Initialize from existing multilingual models
python training/bootstrap_from_pretrained.py

# Fine-tune on your data
python training/train_universal_system.py \
    --encoder_checkpoint models/encoder_pretrained.pt \
    --decoder_checkpoint models/decoder_pretrained.pt \
    --data_dir data/processed \
    --num_epochs 10
```

## Training Configuration

**Small Model (Testing)**
```yaml
# configs/training_small.yaml
model:
  encoder_layers: 4
  decoder_layers: 4
  hidden_dim: 256
  batch_size: 16

training:
  num_epochs: 5
  learning_rate: 5e-5
  warmup_steps: 1000
```

**Full Model**
```yaml
# configs/training_full.yaml
model:
  encoder_layers: 6
  decoder_layers: 6
  hidden_dim: 384
  batch_size: 32

training:
  num_epochs: 20
  learning_rate: 3e-4
  warmup_steps: 4000
```

## Monitoring Training

**View training logs**
```bash
tensorboard --logdir logs/
```

**Monitor GPU usage**
```bash
watch -n 1 nvidia-smi
```

**Checkpointing**
```
Checkpoints are saved during training at regular intervals. The best-performing checkpoint is saved at the end.
```

## Common Issues

**Out of Memory**
- Reduce batch_size in config
- Enable gradient_checkpointing
- Use mixed precision training

**Slow Training**
- Ensure CUDA is properly installed
- Use DataLoader with num_workers > 0
- Consider distributed training

## Model Conversion

**After training, convert models for deployment**:
```bash
# Convert encoder to ONNX
python training/convert_to_onnx.py \
    --checkpoint checkpoints/best_encoder.pt \
    --output models/encoder.onnx

# Optimize for mobile
python training/optimize_for_mobile.py \
    --input models/encoder.onnx \
    --output models/encoder_mobile.onnx
```