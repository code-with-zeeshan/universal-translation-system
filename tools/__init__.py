# tools/__init__.py
"""
Tools and utilities for the Universal Translation System.

Includes CLI helpers like cloud_preflight, prefetch_artifacts, register_decoder_node.

Note: We intentionally avoid importing CLI modules here to prevent side effects
(e.g., argparse) during package import. Import them directly if needed.
"""

__all__: list[str] = []
