"""
Evaluation module for the Universal Translation System.

Provides comprehensive evaluation tools including BLEU, COMET,
and other metrics for translation quality assessment.
"""

from .evaluator import TranslationEvaluator, evaluate_translation_quality
from .evaluator import TranslationPair

__all__ = [
    "TranslationEvaluator",
    "TranslationPair",
    "evaluate_translation_quality",
]
