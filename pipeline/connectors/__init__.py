# pipeline/connectors/__init__.py
"""
Connector package exposing integration connectors used by data and training.
"""
from .data import PipelineConnector
from .vocabulary import VocabularyConnector

__all__ = [
    "PipelineConnector",
    "VocabularyConnector",
]