# __init__.py (root)
"""
Universal Translation System

A state-of-the-art multilingual translation system with dynamic vocabulary loading,
language adapters, and edge deployment capabilities.
"""

__version__ = "1.0.0"
__author__ = "Universal Translation Team"

# Make key components easily importable
from encoder.universal_encoder import UniversalEncoder
from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
from integration.connect_all_systems import UniversalTranslationSystem

__all__ = [
    "UniversalEncoder",
    "OptimizedUniversalDecoder", 
    "UniversalTranslationSystem"
]