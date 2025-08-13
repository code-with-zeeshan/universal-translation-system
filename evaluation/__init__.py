# evaluation/__init__.py
"""
Evaluation module for the Universal Translation System.

Provides comprehensive evaluation tools including BLEU, COMET,
and other metrics for translation quality assessment.
"""

from .evaluate_model import (
    TranslationEvaluator,
    TranslationPair,
    evaluate_translation_quality
)

__all__ = [
    "TranslationEvaluator",
    "TranslationPair",
    "evaluate_translation_quality"
]