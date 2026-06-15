# __init__.py (root)
"""
Universal Translation System

A state-of-the-art multilingual translation system with dynamic vocabulary loading,
language adapters, and edge deployment capabilities.
"""

__version__ = "1.0.0"
__author__ = "Universal Translation Team"

# These are heavy imports (torch, etc.) — deferred via __getattr__
# so that importing the package doesn't force all deps to be installed.
__all__ = [
    "UniversalEncoder",
    "OptimizedUniversalDecoder",
    "UniversalTranslationSystem",
]


def __getattr__(name):
    if name == "UniversalEncoder":
        from runtime.encoder.universal_encoder import UniversalEncoder

        return UniversalEncoder
    if name == "OptimizedUniversalDecoder":
        from runtime.cloud_decoder import OptimizedUniversalDecoder

        return OptimizedUniversalDecoder
    if name == "UniversalTranslationSystem":
        from integration.system import UniversalTranslationSystem

        return UniversalTranslationSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")