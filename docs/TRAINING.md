# Training Guide

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space
- 16GB+ RAM

## Data Preparation
- Download and preprocess data using scripts in `data/`
- Create vocabulary packs with `vocabulary/create_vocabulary_packs_from_data.py`

## Model Training

### Config Auto-Detection
- Training scripts auto-detect GPU type and load the best config (A100, V100, 3090, T4, or fallback)
- Configs are in `config/` and include both data and training parameters

### Example
```bash
python training/train_universal_system.py
# (No need to specify config; it is auto-selected)
```

### Distributed & Memory-Efficient Training
- Use `training/distributed_train.py` for multi-GPU
- Use `training/memory_efficient_training.py` for large models
- All configs support gradient checkpointing, mixed precision, and dynamic batch sizing

### Monitoring
- Logs: `logs/`
- TensorBoard: `tensorboard --logdir logs/`
- GPU: `watch -n 1 nvidia-smi`

### Model Conversion
- Convert trained models for deployment using `training/convert_models.py`

### Troubleshooting
- Out of memory: reduce batch size, enable gradient checkpointing, use mixed precision
- Slow training: check CUDA, use DataLoader with workers, try distributed
- See `docs/TROUBLESHOOT.md` for more