# tools/__init__.py
"""
Tools and utilities for the Universal Translation System.

- Re-exports vocabulary pack creator for convenience
- Includes CLI helpers like cloud_preflight, prefetch_artifacts, register_decoder_node

Note: We intentionally avoid importing CLI modules here to prevent side effects
(e.g., argparse) during package import. Import them directly if needed.
"""

# Re-export the unified vocabulary pack creator from vocabulary package
from vocabulary import VocabularyPackCreator

__all__ = [
    "VocabularyPackCreator",
]