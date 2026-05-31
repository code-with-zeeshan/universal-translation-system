import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for memory optimization"""
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    activation_offload: bool = False
    use_flash_attention: bool = True
    dynamic_batch_size: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    compile_model: bool = True
    empty_cache_freq: int = 100
    max_split_size: int = 512  # MB
    use_fused_optimizer: bool = True
    enable_nested_tensor: bool = True
    use_channels_last: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    use_inductor: bool = True
    dtype: torch.dtype = torch.bfloat16
    profile_memory: bool = False
    use_safetensors: bool = True
