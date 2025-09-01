# encoder/__init__.py
"""
Encoder module for the Universal Translation System.

Provides the universal encoder implementation with support for
language adapters and dynamic vocabulary loading.
"""

from .universal_encoder import UniversalEncoder
from .language_adapters import (
    LanguageAdapter,
    AdapterUniversalEncoder
)
# AdapterManager is not implemented; keep import optional for backward compat
try:
    from .adapter_manager import AdapterManager  # type: ignore
except Exception:
    AdapterManager = None  # type: ignore
from .train_adapters import AdapterTrainer

__all__ = [
    "UniversalEncoder",
    "LanguageAdapter",
    "AdapterUniversalEncoder",
    "AdapterTrainer",
]
# Expose AdapterManager only if available
if AdapterManager is not None:  # type: ignore
    __all__.append("AdapterManager")