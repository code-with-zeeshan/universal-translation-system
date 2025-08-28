# training/intelligent_trainer.py
"""
Intelligent Universal Trainer that automatically adapts to hardware and requirements.
Consolidates single-GPU, multi-GPU, distributed, and memory-efficient training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, BackwardPrefetch, MixedPrecision, ShardingStrategy
)

import os
import sys
import time
import json
import logging
import socket
import wandb
from collections import defaultdict
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Union
from contextlib import contextmanager

# Import existing modules
from training.memory_efficient_training import (
    MemoryOptimizedTrainer, 
    MemoryConfig,
    DynamicBatchSizer,
    MemoryTracker
)
from training.training_utils import (
    check_convergence,
    create_training_report,
    get_optimal_batch_size
)
from utils.gpu_utils import optimize_gpu_memory, get_gpu_memory_info
from utils.resource_monitor import resource_monitor
from utils.shutdown_handler import GracefulShutdown
from config.schemas import RootConfig

# Dataset imports
from data.dataset_classes import ModernParallelDataset
from data.custom_samplers import TemperatureSampler  # If you have this file

# Profiling (optional - use when needed)
from training.profiling import ProfileGuidedTrainer

# Enhanced utilities
from training.training_utils import (
    create_optimizer_with_param_groups,
    calculate_gradient_norm,
    get_training_diagnostics,
    get_adaptive_gradient_clipping_value,
    # ... other utilities as needed
)

logger = logging.getLogger(__name__)

# ==================== Hardware Detection ====================

class HardwareProfile(Enum):
    """Hardware profiles for optimization"""
    # Single GPU profiles
    LOW_END_SINGLE = auto()      # T4, RTX 3060, V100
    MID_RANGE_SINGLE = auto()     # RTX 4090, RTX 3090, A10
    HIGH_END_SINGLE = auto()      # A100, H100 single
    
    # Multi-GPU profiles  
    LOW_END_MULTI = auto()        # Multiple T4s, V100s
    MID_RANGE_MULTI = auto()      # Multiple RTX 4090s
    HIGH_END_MULTI = auto()       # Multiple A100s/H100s
    
    # Special cases
    CPU_ONLY = auto()
    TPU = auto()
    APPLE_SILICON = auto()       # M1/M2/M3

class TrainingAnalytics:
    """Comprehensive training analytics integrated into IntelligentTrainer"""
    
    def __init__(self, training_history: Dict):
        self.metrics = defaultdict(list)
        self.training_history = training_history  # Reference to trainer's history
        self.start_time = time.time()
    
    def log_step(self, loss: float, lr: float, grad_norm: float, 
                 memory_gb: float, tokens_per_sec: float):
        """Log training step metrics"""
        self.metrics['loss'].append(loss)
        self.metrics['lr'].append(lr)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['memory_gb'].append(memory_gb)
        self.metrics['tokens_per_sec'].append(tokens_per_sec)
        self.metrics['timestamp'].append(time.time() - self.start_time)
        
        # Also update main training history
        self.training_history['gradient_norms'].append(grad_norm)
        self.training_history['memory_snapshots'].append(memory_gb)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate training analytics report"""
        return {
            'duration_hours': (time.time() - self.start_time) / 3600,
            'avg_tokens_per_sec': np.mean(self.metrics['tokens_per_sec']) if self.metrics['tokens_per_sec'] else 0,
            'peak_memory_gb': max(self.metrics['memory_gb']) if self.metrics['memory_gb'] else 0,
            'final_loss': self.metrics['loss'][-1] if self.metrics['loss'] else None,
            'loss_reduction': (
                (self.metrics['loss'][0] - self.metrics['loss'][-1]) / self.metrics['loss'][0] 
                if len(self.metrics['loss']) > 1 else 0
            ),
            'total_steps': len(self.metrics['loss']),
            'gradient_norm_stats': {
                'mean': np.mean(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
                'std': np.std(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
                'max': max(self.metrics['grad_norm']) if self.metrics['grad_norm'] else 0,
            }
        }
    
@dataclass
class TrainingStrategy:
    """Optimal training strategy based on hardware"""
    hardware_profile: HardwareProfile
    use_distributed: bool
    distributed_backend: str  # 'nccl', 'gloo', 'mps'
    use_fsdp: bool
    use_ddp: bool
    memory_config: MemoryConfig
    batch_size: int
    accumulation_steps: int
    compile_mode: str
    mixed_precision_dtype: torch.dtype
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    gradient_clipping: float
    learning_rate: float
    warmup_steps: int

# ==================== Main Intelligent Trainer ====================

class IntelligentTrainer:
    """
    Intelligent trainer that automatically adapts to any hardware configuration
    and training scenario. Acts as the brain of the training system.
    """
    
    def __init__(self, 
                 encoder: nn.Module,
                 decoder: nn.Module,
                 train_dataset,
                 val_dataset,
                 config: RootConfig,
                 experiment_name: str = "intelligent-universal",
                 resume_from_checkpoint: Optional[str] = None):
        
        # Core components
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.experiment_name = experiment_name
        
        # Hardware detection and strategy
        self.hardware_info = self._detect_hardware()
        self.hardware_profile = self._categorize_hardware(self.hardware_info)
        self.strategy = self._determine_optimal_strategy()
        
        # Log hardware and strategy
        self._log_initialization()
        
        # Setup device and distributed environment
        self.device, self.local_rank, self.world_size = self._setup_device()
        
        # Initialize components based on strategy
        self._initialize_components()
        
        # Setup models with optimal configuration
        self._setup_models()
        
        # Create optimizers and schedulers
        self._setup_optimization()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Load checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Add analytics tracker
        self.analytics = TrainingAnalytics(self.training_history)
        
        # QAT support
        self.qat_enabled = False
        self.qat_bits = 8    
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'batch_sizes': [],
            'memory_usage': [],
            'step_times': [],
            'gradient_norms': [],
            'weight_updates': [],
            'learning_rate_schedule': [],
            'memory_snapshots': [],
            'layer_wise_stats': defaultdict(list)
        }
    
    # ==================== Hardware Detection Methods ====================
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Comprehensive hardware detection"""
        hardware_info = {
            'cpu_count': os.cpu_count(),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': [],
            'total_memory': 0,
            'backend': 'cpu'
        }
        
        # Check for CUDA GPUs
        if torch.cuda.is_available():
            hardware_info['gpu_count'] = torch.cuda.device_count()
            hardware_info['backend'] = 'cuda'
            
            for i in range(hardware_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                hardware_info['gpu_names'].append(gpu_name)
                hardware_info['gpu_memory'].append(gpu_memory)
                hardware_info['total_memory'] += gpu_memory
                
        # Check for Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            hardware_info['backend'] = 'mps'
            hardware_info['gpu_count'] = 1
            hardware_info['gpu_names'] = ['Apple Silicon GPU']
            
        # Check for TPU
        try:
            import torch_xla
            hardware_info['backend'] = 'xla'
            hardware_info['gpu_count'] = torch_xla.core.xla_model.xrt_world_size()
        except ImportError:
            pass
        
        return hardware_info
    
    def _categorize_hardware(self, hardware_info: Dict[str, Any]) -> HardwareProfile:
        """Categorize hardware into profiles"""
        
        if hardware_info['backend'] == 'cpu':
            return HardwareProfile.CPU_ONLY
            
        if hardware_info['backend'] == 'mps':
            return HardwareProfile.APPLE_SILICON
            
        if hardware_info['backend'] == 'xla':
            return HardwareProfile.TPU
            
        # Categorize CUDA GPUs
        gpu_count = hardware_info['gpu_count']
        if gpu_count == 0:
            return HardwareProfile.CPU_ONLY
            
        # Analyze first GPU as representative
        gpu_name = hardware_info['gpu_names'][0].lower()
        gpu_memory = hardware_info['gpu_memory'][0]
        
        # Categorization rules
        low_end_patterns = ['t4', 'rtx 2060', 'rtx 2070', 'rtx 2080', 'rtx 3060', 'gtx', 'p100', 'k80']
        mid_range_patterns = ['rtx 3070', 'rtx 3080', 'rtx 3090', 'rtx 4070', 'rtx 4080', 'rtx 4090', 'a10', 'v100']
        high_end_patterns = ['a100', 'h100', 'a6000', 'a40', 'a30']
        
        # Check patterns
        is_low_end = any(pattern in gpu_name for pattern in low_end_patterns) or gpu_memory < 16
        is_mid_range = any(pattern in gpu_name for pattern in mid_range_patterns) or (16 <= gpu_memory < 40)
        is_high_end = any(pattern in gpu_name for pattern in high_end_patterns) or gpu_memory >= 40
        
        if gpu_count == 1:
            if is_high_end:
                return HardwareProfile.HIGH_END_SINGLE
            elif is_mid_range:
                return HardwareProfile.MID_RANGE_SINGLE
            else:
                return HardwareProfile.LOW_END_SINGLE
        else:  # Multiple GPUs
            if is_high_end:
                return HardwareProfile.HIGH_END_MULTI
            elif is_mid_range:
                return HardwareProfile.MID_RANGE_MULTI
            else:
                return HardwareProfile.LOW_END_MULTI
    
    def _determine_optimal_strategy(self) -> TrainingStrategy:
        """Determine optimal training strategy based on hardware - using extracted YAML values"""
        
        # Check for BFloat16 support
        bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        # Comprehensive strategy mapping based on your YAML configs
        strategies = {
            # ============ HIGH-END GPUS ============
        
            # H100 (from training_h100.yaml)
            HardwareProfile.HIGH_END_SINGLE: TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=False,  # h100: false
                    mixed_precision=True,
                    cpu_offload=False,  # h100: false
                    activation_offload=False,  # h100: false
                    compile_model=True,
                    dtype="bfloat16",  # h100: bfloat16
                    compile_mode="max-autotune",  # h100: max-autotune
                    use_flash_attention=True,
                    use_channels_last=True,
                    enable_nested_tensor=True,
                    use_inductor=True,
                    empty_cache_freq=100,
                    max_split_size=512
                ),
                batch_size=128,  # h100: 128
                accumulation_steps=1,  # h100: 1
                compile_mode="max-autotune",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=8,
                gradient_clipping=1.0,
                learning_rate=6e-4,  # h100: 6e-4
                warmup_steps=4000  # from base.yaml
            ),
        
            # A100 (from training_a100.yaml) 
            # Also handles as HIGH_END_SINGLE alternative
            "a100_single": TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=False,  # a100: false
                    mixed_precision=True,
                    cpu_offload=False,  # a100: false
                    activation_offload=False,  # a100: false
                    compile_model=True,
                    dtype="bfloat16",  # a100: bfloat16
                    compile_mode="max-autotune",  # a100: max-autotune
                    use_flash_attention=True,
                    empty_cache_freq=100,
                    max_split_size=512
                ),
                batch_size=64,  # a100: 64
                accumulation_steps=2,  # a100: 2
                compile_mode="max-autotune",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=8,
                gradient_clipping=1.0,
                learning_rate=5e-4,  # a100: 5e-4
                warmup_steps=4000
            ),
        
            # A100 FSDP (from training_a100_fsdp.yaml) - for multi-GPU
            HardwareProfile.HIGH_END_MULTI: TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_MULTI,
                use_distributed=True,
                distributed_backend='nccl',
                use_fsdp=True,  # a100_fsdp: true
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # a100_fsdp: true
                    mixed_precision=True,
                    cpu_offload=False,
                    compile_model=True,
                    dtype="bfloat16",  # a100_fsdp: bfloat16
                    compile_mode="max-autotune",  # a100_fsdp: max-autotune
                    use_flash_attention=True,
                    use_inductor=True,
                    use_channels_last=True,
                    enable_nested_tensor=True
                ),
                batch_size=128 * self.hardware_info['gpu_count'],  # a100_fsdp: 128 per GPU
                accumulation_steps=1,
                compile_mode="max-autotune",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=8,
                gradient_clipping=1.0,
                learning_rate=5e-4,
                warmup_steps=4000
            ),
        
            # ============ MID-RANGE GPUS ============
        
            # RTX 4090 (from training_rtx4090.yaml)
            HardwareProfile.MID_RANGE_SINGLE: TrainingStrategy(
                hardware_profile=HardwareProfile.MID_RANGE_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                   gradient_checkpointing=True,  # rtx4090: true
                   mixed_precision=True,
                   cpu_offload=False,  # rtx4090: false
                   activation_offload=False,  # rtx4090: false
                   compile_model=True,
                   dtype="bfloat16",  # rtx4090: bfloat16
                   compile_mode="reduce-overhead",  # rtx4090: reduce-overhead
                   use_flash_attention=True,
                   use_channels_last=True,
                   max_split_size=256,  # rtx4090: 256
                   empty_cache_freq=100
                ),
                batch_size=24,  # rtx4090: 24
                accumulation_steps=5,  # rtx4090: 5
                compile_mode="reduce-overhead",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                gradient_clipping=1.0,
                learning_rate=4e-4,  # rtx4090: 4e-4
                warmup_steps=2000
            ),
        
            # RTX 3090 (from training_rtx3090.yaml)
            "rtx3090": TrainingStrategy(
                hardware_profile=HardwareProfile.MID_RANGE_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # rtx3090: true
                    mixed_precision=True,
                    cpu_offload=True,  # rtx3090: true
                    activation_offload=False,  # rtx3090: false
                    compile_model=True,
                    dtype="bfloat16",  # rtx3090: bfloat16
                    compile_mode="reduce-overhead",  # rtx3090: reduce-overhead
                    use_flash_attention=True,
                    max_split_size=256,  # rtx3090: 256
                    empty_cache_freq=100
                ),
                batch_size=16,  # rtx3090: 16
                accumulation_steps=8,  # rtx3090: 8
                compile_mode="reduce-overhead",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                gradient_clipping=1.0,
                learning_rate=3e-4,  # rtx3090: 3e-4
                warmup_steps=2000
            ),
        
            # V100 (from training_v100.yaml)
            "v100": TrainingStrategy(
                hardware_profile=HardwareProfile.MID_RANGE_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # v100: true
                    mixed_precision=True,
                    cpu_offload=False,  # v100: false
                    activation_offload=False,  # v100: false
                    compile_model=True,
                    dtype="float16",  # v100: float16 (no bfloat16 support)
                    compile_mode="reduce-overhead",  # v100: reduce-overhead
                    use_flash_attention=True,
                    empty_cache_freq=100,
                    max_split_size=512
                ),
                batch_size=32,  # v100: 32
                accumulation_steps=4,  # v100: 4
                compile_mode="reduce-overhead",
                mixed_precision_dtype=torch.float16,  # V100 uses float16
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                gradient_clipping=1.0,
                learning_rate=3e-4,  # v100: 3e-4
                warmup_steps=2000
            ),
        
            # L4 (from training_l4.yaml)
            "l4": TrainingStrategy(
                hardware_profile=HardwareProfile.MID_RANGE_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # l4: true
                    mixed_precision=True,
                    cpu_offload=False,  # l4: false
                    activation_offload=False,  # l4: false
                    compile_model=True,
                    dtype="bfloat16",  # l4: bfloat16
                    compile_mode="default",  # l4: default
                    use_flash_attention=True,
                    max_split_size=256,  # l4: 256
                    empty_cache_freq=100
                ),
                batch_size=16,  # l4: 16
                accumulation_steps=8,  # l4: 8
                compile_mode="default",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=4,
                gradient_clipping=1.0,
                learning_rate=2e-4,  # l4: 2e-4
                warmup_steps=1500
            ),
        
            # ============ LOW-END GPUS ============
        
            # RTX 3080 (from training_rtx3080.yaml)
            HardwareProfile.LOW_END_SINGLE: TrainingStrategy(
                hardware_profile=HardwareProfile.LOW_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # rtx3080: true
                    mixed_precision=True,
                    cpu_offload=True,  # rtx3080: true
                    activation_offload=False,  # rtx3080: false
                    compile_model=True,
                    dtype="bfloat16",  # rtx3080: bfloat16
                    compile_mode="reduce-overhead",  # rtx3080: reduce-overhead
                    use_flash_attention=True,
                    max_split_size=128,  # rtx3080: 128
                    empty_cache_freq=50  # rtx3080: 50
                ),
                batch_size=8,  # rtx3080: 8
                accumulation_steps=16,  # rtx3080: 16
                compile_mode="reduce-overhead",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                gradient_clipping=1.0,
                learning_rate=3e-4,  # rtx3080: 3e-4
                warmup_steps=1000
            ),
        
            # RTX 3060 (from training_rtx3060.yaml)
            "rtx3060": TrainingStrategy(
                hardware_profile=HardwareProfile.LOW_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # rtx3060: true
                    mixed_precision=True,
                    cpu_offload=True,  # rtx3060: true
                    activation_offload=True,  # rtx3060: true
                    compile_model=True,
                    dtype="bfloat16",  # rtx3060: bfloat16
                    compile_mode="reduce-overhead",  # rtx3060: reduce-overhead
                    use_flash_attention=True,
                    max_split_size=128,  # rtx3060: 128
                    empty_cache_freq=50  # rtx3060: 50
                ),
                batch_size=8,  # rtx3060: 8
                accumulation_steps=16,  # rtx3060: 16
                compile_mode="reduce-overhead",
                mixed_precision_dtype=torch.bfloat16,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                gradient_clipping=1.0,
                learning_rate=2e-4,  # rtx3060: 2e-4
                warmup_steps=1000
            ),
         
            # T4 (from training_t4.yaml)
            "t4": TrainingStrategy(
                hardware_profile=HardwareProfile.LOW_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # t4: true
                    mixed_precision=True,
                    cpu_offload=True,  # t4: true
                    activation_offload=True,  # t4: true
                    compile_model=True,
                    dtype="float16",  # t4: float16
                    compile_mode="default",  # t4: default
                    use_flash_attention=False,  # T4 may not support
                    max_split_size=128,  # t4: 128
                    empty_cache_freq=50  # t4: 50
                ),
                batch_size=8,  # t4: 8
                accumulation_steps=16,  # t4: 16
                compile_mode="default",
                mixed_precision_dtype=torch.float16,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                gradient_clipping=1.0,
                learning_rate=2e-4,  # t4: 2e-4
                warmup_steps=500
            ),
        
            # Colab Free Tier (from training_colab_free.yaml - K80/T4)
            "colab_free": TrainingStrategy(
                hardware_profile=HardwareProfile.LOW_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=True,  # colab: true
                    mixed_precision=True,
                    cpu_offload=True,  # colab: true
                    activation_offload=True,  # colab: true
                    compile_model=False,  # colab: false (can be slow)
                    dtype="float16",  # colab: float16
                    compile_mode="default",
                    use_flash_attention=False,  # colab: false (K80 doesn't support)
                    max_split_size=64,  # colab: 64
                    empty_cache_freq=25  # colab: 25
                ),
                batch_size=4,  # colab: 4
                accumulation_steps=32,  # colab: 32
                compile_mode="default",
                mixed_precision_dtype=torch.float16,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                gradient_clipping=1.0,
                learning_rate=1e-4,  # colab: 1e-4
                warmup_steps=500
            ),
        
            # ============ SPECIAL HARDWARE ============
        
            # AMD MI250 (from training_amd_mi250.yaml)
            "amd_mi250": TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_SINGLE,
                use_distributed=False,
                distributed_backend='nccl',  # PyTorch maps to rccl on ROCm
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=False,  # mi250: false
                    mixed_precision=True,
                    cpu_offload=False,  # mi250: false
                    activation_offload=False,  # mi250: false
                    compile_model=True,
                    dtype="float16",  # mi250: float16 (broadly supported on ROCm)
                    compile_mode="max-autotune",  # mi250: max-autotune
                    use_flash_attention=True,  # mi250: true (ROCm 5.6+)
                    empty_cache_freq=100
                ),
                batch_size=64,  # mi250: 64
                accumulation_steps=2,  # mi250: 2
                compile_mode="max-autotune",
                mixed_precision_dtype=torch.float16,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=8,
                gradient_clipping=1.0,
                learning_rate=5e-4,  # mi250: 5e-4
                warmup_steps=4000
            ),
        
            # CPU Only (from training_cpu.yaml)
            HardwareProfile.CPU_ONLY: TrainingStrategy(
                hardware_profile=HardwareProfile.CPU_ONLY,
                use_distributed=False,
                distributed_backend='gloo',  # cpu: gloo
                use_fsdp=False,
                use_ddp=False,
                memory_config=MemoryConfig(
                    gradient_checkpointing=False,  # cpu: false (not effective)
                    mixed_precision=False,  # cpu: must be false
                    cpu_offload=False,  # cpu: irrelevant
                    activation_offload=False,  # cpu: irrelevant
                    compile_model=False,  # cpu: false
                    dtype="float32",  # cpu: float32
                    compile_mode="default",
                    use_flash_attention=False,  # cpu: false
                    dynamic_batch_size=True
                ),
                batch_size=4,  # cpu: 4
                accumulation_steps=1,  # cpu: 1
                compile_mode="default",
                mixed_precision_dtype=torch.float32,
                num_workers=2,
                pin_memory=False,  # cpu: false
                prefetch_factor=2,
                gradient_clipping=1.0,
                learning_rate=1e-4,  # cpu: 1e-4
                warmup_steps=100
            ),
        }
    
        # GPU name mapping for detection
        gpu_mappings = {
           'h100': strategies[HardwareProfile.HIGH_END_SINGLE],
           'a100': strategies["a100_single"],
           'rtx 4090': strategies[HardwareProfile.MID_RANGE_SINGLE],
           'rtx 3090': strategies["rtx3090"],
           'rtx 3080': strategies[HardwareProfile.LOW_END_SINGLE],
           'rtx 3060': strategies["rtx3060"],
           'v100': strategies["v100"],
           'l4': strategies["l4"],
           't4': strategies["t4"],
           'k80': strategies["colab_free"],
           'mi250': strategies["amd_mi250"],
           'mi300': strategies["amd_mi250"],  # Same config as MI250
        }
    
        # Detect GPU and return appropriate strategy
        if self.hardware_info['gpu_names']:
            gpu_name = self.hardware_info['gpu_names'][0].lower()
        
            # Check for exact matches first
            for gpu_key, strategy in gpu_mappings.items():
                if gpu_key in gpu_name:
                    logger.info(f"Matched GPU '{gpu_key}' - using optimized settings")
                    return strategy
        
            # Fallback to memory-based detection
            gpu_memory = self.hardware_info['gpu_memory'][0]
        
            if gpu_memory >= 70:  # 80GB GPUs (H100, A100 80GB)
                return strategies[HardwareProfile.HIGH_END_SINGLE]
            elif gpu_memory >= 40:  # 40-48GB GPUs (A100 40GB, A6000)
                return strategies["a100_single"]
            elif gpu_memory >= 20:  # 24GB GPUs (RTX 3090/4090, L4)
                return strategies[HardwareProfile.MID_RANGE_SINGLE]
            elif gpu_memory >= 12:  # 12-16GB GPUs (RTX 3060, T4, V100)
                return strategies[HardwareProfile.LOW_END_SINGLE]
            else:  # <12GB (Colab free tier)
                return strategies["colab_free"]
    
        # Multi-GPU configurations
        if self.hardware_info['gpu_count'] > 1:
            # Determine multi-GPU profile based on first GPU
            if self.hardware_info['gpu_memory'][0] >= 40:
                return strategies[HardwareProfile.HIGH_END_MULTI]
            else:
                # For mid/low-end multi-GPU, create a DDP strategy
                base_strategy = self._get_single_gpu_strategy()
                base_strategy.use_distributed = True
                base_strategy.use_ddp = True
                base_strategy.batch_size *= self.hardware_info['gpu_count']
                return base_strategy
    
        # CPU fallback
        return strategies[HardwareProfile.CPU_ONLY]
        
        strategy = strategies.get(self.hardware_profile, strategies[HardwareProfile.CPU_ONLY])
        
        # Override with user config if specified
        if hasattr(self.config.training, 'force_batch_size'):
            strategy.batch_size = self.config.training.force_batch_size
        if hasattr(self.config.training, 'force_learning_rate'):
            strategy.learning_rate = self.config.training.force_learning_rate
            
        return strategy
    
    # ==================== Setup Methods ====================
    
    def find_free_port(self):
        """Find a free port for distributed training"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _setup_device(self) -> Tuple[torch.device, int, int]:
        """Setup device and distributed environment"""
        
        if self.strategy.use_distributed and torch.cuda.device_count() > 1:
            # Setup distributed training
            if 'RANK' in os.environ:
                # Already initialized by launcher
                local_rank = int(os.environ['LOCAL_RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                # Initialize here
                local_rank = 0
                world_size = torch.cuda.device_count()
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355' or str(find_free_port())
                
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.strategy.distributed_backend,
                    world_size=world_size,
                    rank=local_rank,
                    timeout=torch.distributed.timedelta(seconds=3600)
                )
                
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            local_rank = 0
            world_size = 1
            
        elif self.hardware_info['backend'] == 'mps':
            device = torch.device('mps')
            local_rank = 0
            world_size = 1
            
        else:
            device = torch.device('cpu')
            local_rank = 0
            world_size = 1
            
        return device, local_rank, world_size
    
    def _initialize_components(self):
        """Initialize training components based on strategy"""
        
        # Memory optimizer
        self.memory_trainer = MemoryOptimizedTrainer(
            self.encoder,  # Will be wrapped later
            self.strategy.memory_config
        )
        
        # Dynamic batch sizer
        self.batch_sizer = DynamicBatchSizer(
            initial_batch_size=self.strategy.batch_size // self.world_size if self.world_size > 1 else self.strategy.batch_size,
            max_batch_size=self.strategy.batch_size * 2 // self.world_size if self.world_size > 1 else self.strategy.batch_size * 2
        )
        
        # Memory tracker
        self.memory_tracker = MemoryTracker() if self.strategy.memory_config.profile_memory else None
        
        # Gradient scaler for mixed precision
        if self.strategy.memory_config.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def _setup_models(self):
        """Setup and wrap models based on strategy"""
        
        # Move models to device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Apply memory optimizations
        if self.strategy.memory_config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        # Apply model compilation if beneficial
        if self.strategy.memory_config.compile_model and hasattr(torch, 'compile'):
            try:
                self.encoder = torch.compile(
                    self.encoder,
                    mode=self.strategy.compile_mode,
                    fullgraph=False,
                    dynamic=True
                )
                self.decoder = torch.compile(
                    self.decoder,
                    mode=self.strategy.compile_mode,
                    fullgraph=False,
                    dynamic=True
                )
                logger.info(f"✅ Models compiled with mode: {self.strategy.compile_mode}")
            except Exception as e:
                logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Apply distributed wrapping
        if self.strategy.use_distributed:
            if self.strategy.use_fsdp:
                self._wrap_models_fsdp()
            elif self.strategy.use_ddp:
                self._wrap_models_ddp()
    
    def _wrap_models_fsdp(self):
        """Wrap models with FSDP"""
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        
        # FSDP configuration
        mixed_precision_policy = None
        if self.strategy.memory_config.mixed_precision:
            mixed_precision_policy = MixedPrecision(
                param_dtype=self.strategy.mixed_precision_dtype,
                reduce_dtype=self.strategy.mixed_precision_dtype,
                buffer_dtype=self.strategy.mixed_precision_dtype,
            )
        
        cpu_offload_policy = CPUOffload(offload_params=True) if self.strategy.memory_config.cpu_offload else None
        
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            }
        )
        
        # Wrap models
        self.encoder = FSDP(
            self.encoder,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.local_rank,
            use_orig_params=True
        )
        
        self.decoder = FSDP(
            self.decoder,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.local_rank,
            use_orig_params=True
        )
        
        logger.info("✅ Models wrapped with FSDP")
    
    def _wrap_models_ddp(self):
        """Wrap models with DDP"""
        self.encoder = DDP(
            self.encoder,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )
        
        self.decoder = DDP(
            self.decoder,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )
        
        logger.info("✅ Models wrapped with DDP")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on models"""
        
        # Try model-specific method first
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable()
            
        logger.info("✅ Gradient checkpointing enabled")
    
    def _setup_optimization(self):
        """Setup optimizers and schedulers"""
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with hardware-specific configuration"""
        
        # Use the enhanced optimizer creation with param groups
        if self.config.training.use_param_groups:
            return create_optimizer_with_param_groups(
                [self.encoder, self.decoder],
                self.config
            )
        else:    
            # Combine parameters (Original optimizer creation)
            parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
            # Use fused optimizer if available and beneficial
            use_fused = (
                self.strategy.memory_config.use_fused_optimizer and 
                torch.cuda.is_available() and
                self.hardware_profile in [
                    HardwareProfile.HIGH_END_SINGLE,
                    HardwareProfile.HIGH_END_MULTI,
                    HardwareProfile.MID_RANGE_SINGLE,
                    HardwareProfile.MID_RANGE_MULTI
                ]
            )
        
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.strategy.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.01,
                fused=use_fused
            )
        
            logger.info(f"✅ Optimizer created (fused={use_fused})")
            return optimizer

    def profile_training(self, num_steps: int = 10):
        """Add profiling capability"""
        profiler = ProfileGuidedTrainer(self)
        return profiler.profile_training_step(num_steps)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get training diagnostics"""
        return get_training_diagnostics(self)        
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        
        # Estimate total training steps
        steps_per_epoch = len(self.train_dataset) // (self.strategy.batch_size * self.strategy.accumulation_steps)
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        # Use OneCycleLR for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.strategy.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        logger.info(f"✅ Scheduler created (total_steps={total_steps})")
        return scheduler
    
    def _setup_monitoring(self):
        """Setup monitoring and logging"""
        
        # Setup WandB if enabled
        if self.config.monitoring.use_wandb and (self.local_rank == 0 or not self.strategy.use_distributed):
            wandb.init(
                project="universal-translation",
                name=f"{self.experiment_name}-{self.hardware_profile.name}",
                config={
                    'hardware': self.hardware_info,
                    'hardware_profile': self.hardware_profile.name,
                    'strategy': {
                        'batch_size': self.strategy.batch_size,
                        'accumulation_steps': self.strategy.accumulation_steps,
                        'learning_rate': self.strategy.learning_rate,
                        'mixed_precision': str(self.strategy.mixed_precision_dtype),
                        'distributed': self.strategy.use_distributed,
                        'backend': self.strategy.distributed_backend
                    },
                    'config': self.config.dict()
                },
                tags=[
                    'intelligent-trainer',
                    self.hardware_profile.name,
                    'distributed' if self.strategy.use_distributed else 'single-gpu'
                ]
            )
            
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_initialization(self):
        """Log initialization details"""
        
        logger.info("="*60)
        logger.info("🧠 INTELLIGENT TRAINER INITIALIZED")
        logger.info("="*60)
        logger.info(f"📊 Hardware Detection:")
        logger.info(f"   - Backend: {self.hardware_info['backend']}")
        logger.info(f"   - GPUs: {self.hardware_info['gpu_count']}")
        if self.hardware_info['gpu_names']:
            for i, (name, mem) in enumerate(zip(self.hardware_info['gpu_names'], self.hardware_info['gpu_memory'])):
                logger.info(f"   - GPU {i}: {name} ({mem:.1f}GB)")
        logger.info(f"   - Total GPU Memory: {self.hardware_info['total_memory']:.1f}GB")
        logger.info(f"   - CPU Cores: {self.hardware_info['cpu_count']}")
        
        logger.info(f"\n🎯 Hardware Profile: {self.hardware_profile.name}")
        
        logger.info(f"\n⚙️ Training Strategy:")
        logger.info(f"   - Distributed: {self.strategy.use_distributed}")
        logger.info(f"   - Backend: {self.strategy.distributed_backend}")
        logger.info(f"   - FSDP: {self.strategy.use_fsdp}")
        logger.info(f"   - DDP: {self.strategy.use_ddp}")
        logger.info(f"   - Batch Size: {self.strategy.batch_size}")
        logger.info(f"   - Accumulation Steps: {self.strategy.accumulation_steps}")
        logger.info(f"   - Learning Rate: {self.strategy.learning_rate}")
        logger.info(f"   - Mixed Precision: {self.strategy.mixed_precision_dtype}")
        logger.info(f"   - Compile Mode: {self.strategy.compile_mode}")
        logger.info(f"   - Gradient Checkpointing: {self.strategy.memory_config.gradient_checkpointing}")
        logger.info(f"   - Flash Attention: {self.strategy.memory_config.use_flash_attention}")
        logger.info("="*60)
    
    # ==================== Training Methods ====================
    
    def train(self, 
             num_epochs: Optional[int] = None,
             shutdown_handler: Optional[GracefulShutdown] = None) -> Dict[str, Any]:
        """
        Main training loop with intelligent adaptation
        """
        
        num_epochs = num_epochs or self.config.training.num_epochs
        
        logger.info(f"🚀 Starting intelligent training for {num_epochs} epochs")
        
        # Create data loaders
        train_loader = self._create_train_loader()
        val_loader = self._create_val_loader()
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Check for shutdown
            if shutdown_handler and shutdown_handler.should_stop():
                logger.info("🛑 Shutdown requested, saving checkpoint...")
                self.save_checkpoint(is_emergency=True)
                break
            
            # Train epoch
            with resource_monitor.monitor(f"epoch_{epoch}"):
                train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            self.training_history['batch_sizes'].append(self.batch_sizer.current_batch_size)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                
            # Regular checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint()
            
            # Adaptive adjustments
            self._adaptive_adjustments(train_metrics, val_metrics)
            
            # Recreate data loader if batch size changed
            if self.batch_sizer.current_batch_size != train_loader.batch_size:
                logger.info(f"📊 Adjusting batch size: {train_loader.batch_size} -> {self.batch_sizer.current_batch_size}")
                train_loader = self._create_train_loader()
        
        # Final report
        self._generate_final_report()
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward and backward pass
            loss = self._training_step(batch)
            
            # Accumulate loss
            epoch_loss += loss
            num_batches += 1
            
            # Log step time
            step_time = time.time() - step_start
            self.training_history['step_times'].append(step_time)
            
            # Log periodically
            if batch_idx % self.config.training.log_every == 0:
                self._log_step(batch_idx, len(train_loader), loss, step_time)
            
            self.global_step += 1
            
            # Memory management
            if self.memory_tracker:
                self.memory_tracker.log_memory_usage()
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'batches': num_batches
        }

    def enable_quantization_aware_training(self, num_bits: int = 8):
        """Enable QAT during training for better quantization robustness"""
        self.qat_enabled = True
        self.qat_bits = num_bits
        logger.info(f"✅ Quantization-aware training enabled with {num_bits} bits")
    
    def _apply_fake_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization for QAT"""
        if not self.qat_enabled or not self.training:
            return tensor
        
        # Calculate quantization parameters
        qmin = -(2 ** (self.qat_bits - 1))
        qmax = (2 ** (self.qat_bits - 1)) - 1
        
        # Get scale and zero point
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - torch.round(tensor.min() / scale)
        
        # Quantize and dequantize
        tensor_q = torch.round(tensor / scale + zero_point)
        tensor_q = torch.clamp(tensor_q, qmin, qmax)
        tensor_dq = (tensor_q - zero_point) * scale
        
        # Straight-through estimator
        return tensor + (tensor_dq - tensor).detach()   
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with accumulation
           Enhanced training step with QAT support
        """
        
        # Determine if we should sync gradients
        should_sync = (self.global_step + 1) % self.strategy.accumulation_steps == 0
        
        # Context for gradient sync control
        if self.strategy.use_distributed and not should_sync:
            # Disable gradient sync for accumulation steps
            sync_context = self._no_sync_context()
        else:
            sync_context = contextmanager(lambda: iter([None]))()
        
        with sync_context:
            # Forward pass
            if self.strategy.memory_config.mixed_precision:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.strategy.mixed_precision_dtype):
                    loss = self._compute_loss(batch)
                    loss = loss / self.strategy.accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(batch)
                loss = loss / self.strategy.accumulation_steps
                loss.backward()
        
        # Update weights if accumulation complete
        if should_sync:
            if self.strategy.memory_config.mixed_precision:
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.strategy.gradient_clipping
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.strategy.gradient_clipping
                )

        # Apply QAT if enabled
        if self.qat_enabled:
            # Apply fake quantization to model weights
            for param in self.encoder.parameters():
                param.data = self._apply_fake_quantization(param.data)
            for param in self.decoder.parameters():
                param.data = self._apply_fake_quantization(param.data)    
                
                # Optimizer step
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

        # Log analytics
        if hasattr(self, 'analytics'):
            grad_norm = calculate_gradient_norm([self.encoder, self.decoder])
            memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            tokens_per_sec = self.batch_sizer.current_batch_size * 512 / step_time  # Approximate
            
            self.analytics.log_step(
                loss=loss.item(),
                lr=self.scheduler.get_last_lr()[0],
                grad_norm=grad_norm,
                memory_gb=memory_gb,
                tokens_per_sec=tokens_per_sec
            )    
        
        return loss.item() * self.strategy.accumulation_steps
    
    @torch.inference_mode()
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.encoder.eval()
        self.decoder.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch in val_loader:
            batch = self._prepare_batch(batch)
            
            if self.strategy.memory_config.mixed_precision:
                with torch.amp.autocast(device_type=self.device.type, dtype=self.strategy.mixed_precision_dtype):
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'batches': num_batches
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch"""
        
        source_ids = batch['source_ids']
        target_ids = batch['target_ids']
        source_mask = batch['source_mask']
        
        # Encoder forward
        encoder_output = self.encoder(source_ids, source_mask)
        
        # Decoder forward
        decoder_output = self.decoder(
            target_ids[:, :-1],
            encoder_output,
            encoder_attention_mask=source_mask
        )
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=batch.get('pad_token_id', 0),
            label_smoothing=0.1 # Same label smoothing
        )
        
        return loss
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for training"""
        
        # Move tensors to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
                
                # Convert to channels last if applicable
                if self.strategy.memory_config.use_channels_last and value.dim() == 4:
                    batch[key] = batch[key].to(memory_format=torch.channels_last)
        
        return batch
    
    @contextmanager
    def _no_sync_context(self):
        """Context manager for disabling gradient sync in distributed training"""
        if self.strategy.use_ddp:
            with self.encoder.no_sync(), self.decoder.no_sync():
                yield
        else:
            yield
    
    # ==================== Data Loading ====================
    
    def _create_train_loader(self) -> DataLoader:
        """Create optimized training data loader"""
        
        # Create sampler
        if self.config.training.use_temperature_sampling:
            from data.custom_samplers import TemperatureSampler
            sampler = TemperatureSampler(
                self.train_dataset,
                batch_size=self.batch_sizer.current_batch_size,
                temperature=self.config.training.temperature
            )
        elif self.strategy.use_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
                seed=42
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Create loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_sizer.current_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.strategy.num_workers,
            pin_memory=self.strategy.pin_memory,
            prefetch_factor=self.strategy.prefetch_factor if self.strategy.num_workers > 0 else None,
            persistent_workers=self.strategy.num_workers > 0,
            drop_last=True
        )
        
        return train_loader
    
    def _create_val_loader(self) -> DataLoader:
        """Create validation data loader"""
        
        # Create sampler
        if self.strategy.use_distributed:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            sampler = None
        
        # Create loader with larger batch size (no gradients)
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_sizer.current_batch_size * 2,
            shuffle=False,
            sampler=sampler,
            num_workers=self.strategy.num_workers,
            pin_memory=self.strategy.pin_memory,
            drop_last=False
        )
        
        return val_loader
    
    # ==================== Adaptive Methods ====================
    
    def _adaptive_adjustments(self, train_metrics: Dict, val_metrics: Dict):
        """Make adaptive adjustments based on metrics"""
        
        # Check for memory pressure
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if memory_used > 0.95:
                # Critical memory usage
                new_batch_size = self.batch_sizer.decrease_batch_size()
                logger.warning(f"⚠️ Critical memory usage ({memory_used:.1%}), reducing batch size to {new_batch_size}")
                
                # Enable more aggressive memory saving
                if not self.strategy.memory_config.gradient_checkpointing:
                    self.strategy.memory_config.gradient_checkpointing = True
                    self._enable_gradient_checkpointing()
                    logger.info("✅ Enabled gradient checkpointing due to memory pressure")
                    
            elif memory_used < 0.7 and len(self.training_history['val_loss']) > 3:
                # Check if we can increase batch size
                recent_losses = self.training_history['val_loss'][-3:]
                if all(recent_losses[i] > recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    # Model is improving, try larger batch
                    new_batch_size = self.batch_sizer.increase_batch_size()
                    logger.info(f"📈 Memory usage low ({memory_used:.1%}) and model improving, increasing batch size to {new_batch_size}")
        
        # Check for convergence
        if check_convergence(self.training_history['train_loss']):
            logger.info("📊 Training appears to have converged")
            
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            logger.info("🔽 Reduced learning rate by 50%")
    
    # ==================== Logging Methods ====================
    
    def _log_step(self, batch_idx: int, total_batches: int, loss: float, step_time: float):
        """Log training step"""
        
        if self.local_rank == 0 or not self.strategy.use_distributed:
            tokens_per_sec = (self.batch_sizer.current_batch_size * 512) / step_time  # Approximate
            
            logger.info(
                f"Step {batch_idx}/{total_batches} | "
                f"Loss: {loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                f"Batch: {self.batch_sizer.current_batch_size} | "
                f"Time: {step_time:.2f}s | "
                f"Tokens/s: {tokens_per_sec:.0f}"
            )
            
            if self.config.monitoring.use_wandb:
                wandb.log({
                    'train/loss': loss,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/batch_size': self.batch_sizer.current_batch_size,
                    'train/step_time': step_time,
                    'train/tokens_per_second': tokens_per_sec,
                    'train/global_step': self.global_step
                })
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics"""
        
        if self.local_rank == 0 or not self.strategy.use_distributed:
            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Time: {train_metrics['time']:.1f}s"
            )
            
            if self.config.monitoring.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/val_loss': val_metrics['loss'],
                    'epoch/train_time': train_metrics['time'],
                    'epoch/val_time': val_metrics['time'],
                    'epoch/learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch/batch_size': self.batch_sizer.current_batch_size
                })
    
    # ==================== Checkpoint Methods ====================
    
    def save_checkpoint(self, is_best: bool = False, is_emergency: bool = False):
        """Save training checkpoint"""
        
        if self.local_rank != 0 and self.strategy.use_distributed:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'hardware_profile': self.hardware_profile.name,
            'strategy': {
                'batch_size': self.strategy.batch_size,
                'accumulation_steps': self.strategy.accumulation_steps,
                'learning_rate': self.strategy.learning_rate
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'batch_sizer_state': {
                'current_batch_size': self.batch_sizer.current_batch_size,
                'max_batch_size': self.batch_sizer.max_batch_size,
                'min_batch_size': self.batch_sizer.min_batch_size
            }
        }
        
        # Handle model state dict based on wrapping
        if self.strategy.use_fsdp:
            # FSDP state dict handling
            # Would need proper FSDP state dict extraction
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        elif self.strategy.use_ddp:
            checkpoint['encoder_state_dict'] = self.encoder.module.state_dict()
            checkpoint['decoder_state_dict'] = self.decoder.module.state_dict()
        else:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        
        # Save scaler state if using mixed precision
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Determine filename
        if is_emergency:
            filename = 'emergency_checkpoint.pt'
        elif is_best:
            filename = 'best_model.pt'
        else:
            filename = f'checkpoint_epoch_{self.current_epoch}.pt'
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        if self.strategy.use_ddp:
            self.encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler if available
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        # Load batch sizer state
        if 'batch_sizer_state' in checkpoint:
            self.batch_sizer.current_batch_size = checkpoint['batch_sizer_state']['current_batch_size']
        
        logger.info(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
    
    def _generate_final_report(self):
        """Generate final training report"""
        
        report = {
            'hardware_profile': self.hardware_profile.name,
            'hardware_info': self.hardware_info,
            'strategy_used': {
                'distributed': self.strategy.use_distributed,
                'backend': self.strategy.distributed_backend,
                'batch_size': self.strategy.batch_size,
                'accumulation_steps': self.strategy.accumulation_steps,
                'mixed_precision': str(self.strategy.mixed_precision_dtype)
            },
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
                'total_epochs': self.current_epoch,
                'total_steps': self.global_step
            },
            'training_history': self.training_history,
            'resource_usage': resource_monitor.get_summary()
        }
        
        # Save report
        report_path = self.checkpoint_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Training report saved to {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Hardware Profile: {self.hardware_profile.name}")
        logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info(f"Total Epochs: {self.current_epoch}")
        logger.info(f"Total Steps: {self.global_step}")
        logger.info("="*60)


# ==================== Main Training Function ====================

def train_intelligent(
    encoder: nn.Module,
    decoder: nn.Module,
    train_dataset,
    val_dataset,
    config: RootConfig,
    experiment_name: str = "intelligent-universal",
    resume_from: Optional[str] = None,
    shutdown_handler: Optional[GracefulShutdown] = None
) -> Dict[str, Any]:
    """
    Main entry point for intelligent training
    
    This automatically detects hardware and runs optimal training strategy.
    """
    
    # Initialize intelligent trainer
    trainer = IntelligentTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        experiment_name=experiment_name,
        resume_from_checkpoint=resume_from
    )
    
    # Run training
    results = trainer.train(
        num_epochs=config.training.num_epochs,
        shutdown_handler=shutdown_handler
    )
    
    return results


# ==================== Distributed Launcher ====================

def launch_distributed_intelligent_training(
    rank: int,
    world_size: int,
    encoder: nn.Module,
    decoder: nn.Module,
    train_dataset,
    val_dataset,
    config: RootConfig,
    experiment_name: str
):
    """
    Function to be called by torch.multiprocessing.spawn for distributed training
    """
    
    # Set environment variables for this process
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Run intelligent training
    train_intelligent(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        experiment_name=experiment_name
    )


if __name__ == "__main__":
    import argparse
    from utils.logging_config import setup_logging
    from config.schemas import RootConfig, DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig

    parser = argparse.ArgumentParser(description="Intelligent Trainer Entry Point")
    parser.add_argument("--config", type=str, help="Path to YAML config or 'dynamic'")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic config generation")
    parser.add_argument("--experiment-name", type=str, default="intelligent-universal")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    setup_logging(log_dir="logs/intelligent_training", log_level="INFO")

    # Build config
    use_dynamic = args.dynamic or (args.config and args.config.strip().lower() == "dynamic")
    if use_dynamic:
        # Minimal dynamic defaults; trainer refines strategy internally
        dynamic_cfg = RootConfig(
            data=DataConfig(training_distribution={}),
            model=ModelConfig(),
            training=TrainingConfig(),
            memory=MemoryConfig(),
            vocabulary=VocabularyConfig()
        )
        cfg = dynamic_cfg
    else:
        from config.schemas import load_config
        cfg = load_config(args.config or "config/base.yaml")

    # Device override
    if args.device:
        # training/launch.py consumes device via config; mirror simple behavior by setting env hint
        os.environ["UTS_FORCE_DEVICE"] = args.device

    # Initialize lightweight models and data via launch helpers to avoid duplication
    from training.launch import initialize_models, load_datasets

    encoder, decoder = initialize_models(cfg)
    train_dataset, val_dataset = load_datasets(cfg)

    # Start training
    shutdown_handler = GracefulShutdown()
    results = train_intelligent(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        experiment_name=args.experiment_name,
        resume_from=args.checkpoint,
        shutdown_handler=shutdown_handler
    )

    logger.info("Intelligent training completed")