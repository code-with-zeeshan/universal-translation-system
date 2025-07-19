# encoder/__init__.py
"""
Encoder module for the Universal Translation System.

Provides the universal encoder implementation with support for
language adapters and dynamic vocabulary loading.
"""

from .universal_encoder import UniversalEncoder
from .language_adapters import (
    LanguageAdapter,
    AdapterUniversalEncoder,
    AdapterManager
)
from .train_adapters import AdapterTrainer

__all__ = [
    "UniversalEncoder",
    "LanguageAdapter",
    "AdapterUniversalEncoder",
    "AdapterManager",
    "AdapterTrainer"
]