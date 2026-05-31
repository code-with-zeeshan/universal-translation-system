"""
Comprehensive evaluation module for the Universal Translation System
Supports BLEU, COMET, and custom metrics
"""

import logging

from evaluation.metrics import TranslationPair
from evaluation.evaluator import TranslationEvaluator, evaluate_translation_quality

logger = logging.getLogger(__name__)

__all__ = [
    "TranslationEvaluator",
    "TranslationPair",
    "evaluate_translation_quality",
]

if __name__ == "__main__":
    # Example usage
    logger.info("Translation Evaluation Module")
    logger.info("This module requires initialized encoder/decoder models and test data.")
    logger.info("\nExample usage:")
    logger.info("evaluator = TranslationEvaluator(encoder, decoder, vocab_manager)")
    logger.info("metrics = evaluator.evaluate_file('test_data.tsv')")
