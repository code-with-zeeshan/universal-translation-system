# integration/__init__.py
"""
Integration module for the Universal Translation System.

Connects all components and provides a unified interface for
the complete translation system.
"""

from .connect_all_systems import (
    UniversalTranslationSystem,
    SystemConfig,
    integrate_full_pipeline
)

__all__ = [
    "UniversalTranslationSystem",
    "SystemConfig",
    "integrate_full_pipeline"
]