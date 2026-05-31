# integration/connect_all_systems.py
"""
Re-export shim for the Universal Translation System integration.

Imports and re-exports all public components from the split modules.
"""

# Import translation_api first to trigger method patching on UniversalTranslationSystem
from . import translation_api  # noqa: F401  # patches UniversalTranslationSystem

from .system_config import SystemConfig
from .system_health import SystemHealthMonitor
from .system import UniversalTranslationSystem
from .translation_api import integrate_full_pipeline, integrate_full_pipeline_async

__all__ = [
    "SystemConfig",
    "SystemHealthMonitor",
    "UniversalTranslationSystem",
    "integrate_full_pipeline",
    "integrate_full_pipeline_async",
]
