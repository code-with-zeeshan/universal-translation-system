# training/distributed_train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class DistributedTrainer:
    def __init__(self, gpu_id, world_size):
        self.gpu_id = gpu_id
        self.world_size = world_size
        
        # Initialize distributed training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=gpu_id
        )
        
        # Set device
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
    def setup_model(self, encoder, decoder):
        # Move models to GPU
        encoder = encoder.to(self.device)
        decoder = decoder.to(self.device)
        
        # Wrap with DDP
        encoder = DDP(encoder, device_ids=[self.gpu_id])
        decoder = DDP(decoder, device_ids=[self.gpu_id])
        
        return encoder, decoder
    
    def train_step(self, batch):
        # Gradient accumulation for large batches
        accumulation_steps = 4
        
        with torch.cuda.amp.autocast():  # Mixed precision
            loss = self.compute_loss(batch)
            loss = loss / accumulation_steps
        
        # Scale loss for mixed precision
        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()