# cloud_decoder/__init__.py
"""
Cloud decoder module for the Universal Translation System.

Provides optimized decoder implementation for cloud deployment with
continuous batching and GPU optimization.
"""

from .optimized_decoder import (
    OptimizedUniversalDecoder,
    OptimizedDecoderLayer,
    ContinuousBatcher
)

__all__ = [
    "OptimizedUniversalDecoder",
    "OptimizedDecoderLayer",
    "ContinuousBatcher"
]