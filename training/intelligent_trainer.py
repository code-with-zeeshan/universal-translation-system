# training/intelligent_trainer.py
"""
Re-export shim for the Intelligent Universal Trainer subsystem.
All original functionality is preserved in the split modules below.
"""

from training.hardware_profile import HardwareProfile, find_free_port, launch_distributed_intelligent_training
from training.training_analytics import TrainingAnalytics
from training.training_strategy import TrainingStrategy
from training.trainer import IntelligentTrainer, train_intelligent

__all__ = [
    "IntelligentTrainer",
    "HardwareProfile",
    "TrainingAnalytics",
    "TrainingStrategy",
    "train_intelligent",
    "launch_distributed_intelligent_training",
    "find_free_port",
]
