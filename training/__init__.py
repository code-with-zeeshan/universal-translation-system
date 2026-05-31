# training/__init__.py
"""
Training module for the Universal Translation System.

Provides various training strategies including progressive training,
distributed training, memory-efficient training, and quantization.
"""

from .intelligent_trainer import IntelligentTrainer
from .progressive_training import  ProgressiveTrainingOrchestrator
from .memory_efficient_training import MemoryOptimizedTrainer, MemoryConfig
from .quantization_pipeline import EncoderQuantizer, QualityComparator, ModelProfiler, QualityPreservingQuantizer
from .bootstrap_from_pretrained import PretrainedModelBootstrapper
from .convert_models import ModelConverter

__all__ = [
    "IntelligentTrainer",
    "ProgressiveTrainingOrchestrator",
    "MemoryOptimizedTrainer",
    "MemoryConfig",
    "EncoderQuantizer",
    "QualityComparator",
    "ModelProfiler",
    "QualityPreservingQuantizer",
    "PretrainedModelBootstrapper",
    "ModelConverter"
]
