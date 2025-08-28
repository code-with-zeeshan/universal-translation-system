#!/bin/bash
# scripts/multi-GPU_training.sh

set -e

# For multi-GPU training, auto-detect number of GPUs if not set
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

# Launch distributed training (config auto-detection is built-in)
echo "🚀 Starting distributed training on $NUM_GPUS GPUs..."
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    training.launch train --distributed --config config/training_generic_multi_gpu.yaml

# Monitor GPU usage
echo "🔍 Monitoring GPU usage..."
watch -n 1 nvidia-smi