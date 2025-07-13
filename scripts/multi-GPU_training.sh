# For multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    training/train_universal_system.py \
    --config config/training_a100.yaml \
    --num_epochs 20 \
    --output_dir models/full_model

# Monitor GPU usage
watch -n 1 nvidia-smi