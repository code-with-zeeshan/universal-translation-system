# training/hardware_profile.py
"""
Hardware profile detection for intelligent training adaptation.
"""

import os
import socket
import torch.nn as nn
from enum import Enum, auto
from typing import Optional

from config.schemas import RootConfig


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


def find_free_port():
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


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
    from training.trainer import train_intelligent
    
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
