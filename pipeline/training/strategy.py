# pipeline/training/strategy.py
"""
Training strategy dataclass for hardware-aware training configuration.
"""

import torch
from dataclasses import dataclass

from pipeline.training.hardware import HardwareProfile
from pipeline.training.memory import MemoryConfig


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
