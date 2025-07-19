# training/__init__.py
"""
Training module for the Universal Translation System.

Provides various training strategies including progressive training,
distributed training, memory-efficient training, and quantization.
"""

from .train_universal_system import ModernUniversalSystemTrainer
from .progressive_training import ProgressiveTrainingStrategy
from .distributed_train import UnifiedDistributedTrainer, TrainingConfig
from .memory_efficient_training import MemoryOptimizedTrainer, MemoryConfig
from .quantization_pipeline import EncoderQuantizer, QualityPreservingQuantizer
from .bootstrap_from_pretrained import PretrainedModelBootstrapper
from .convert_models import ModelConverter

__all__ = [
    "ModernUniversalSystemTrainer",
    "ProgressiveTrainingStrategy",
    "UnifiedDistributedTrainer",
    "TrainingConfig",
    "MemoryOptimizedTrainer",
    "MemoryConfig",
    "EncoderQuantizer",
    "QualityPreservingQuantizer",
    "PretrainedModelBootstrapper",
    "ModelConverter"
]