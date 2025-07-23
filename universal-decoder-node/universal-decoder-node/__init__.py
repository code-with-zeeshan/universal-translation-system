# universal-decoder-node/universal_decoder_node/__init__.py
"""
Universal Decoder Node - Cloud decoder service for Universal Translation System.

This package provides a high-performance decoder service that runs on GPU-enabled
servers to handle translation decoding from compressed embeddings.
"""

__version__ = "0.1.0"
__author__ = "Universal Translation System"
__license__ = "Apache 2.0"

from .decoder import (
    OptimizedUniversalDecoder,
    OptimizedDecoderLayer,
    ContinuousBatcher,
    DecoderService
)

__all__ = [
    "OptimizedUniversalDecoder",
    "OptimizedDecoderLayer", 
    "ContinuousBatcher",
    "DecoderService"
]