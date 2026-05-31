# integration/translation_api.py
"""
Translation API with async support for the Universal Translation System
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import yaml
from prometheus_client import Counter, Histogram
from utils.unified_validation import InputValidator
from utils.constants import MODELS_ADAPTERS_DIR, CONFIG_DIR

from .system import UniversalTranslationSystem
from .system_config import SystemConfig

logger = logging.getLogger(__name__)

# Metrics
translation_counter = Counter('translations_total', 'Total translations', ['source_lang', 'target_lang'])
translation_duration = Histogram('translation_duration_seconds', 'Translation duration')


# --- Translate methods (patched onto UniversalTranslationSystem) ---

def translate(self, text: str, source_lang: str, target_lang: str, domain: Optional[str] = None) -> str:
    """Translate text with input validation and optional domain-specific expertise."""
    # Validation:
    text = InputValidator.validate_text_input(text, max_length=5000)

    if not InputValidator.validate_language_code(source_lang):
        raise ValueError(f"Invalid source language: {source_lang}")

    if not InputValidator.validate_language_code(target_lang):
        raise ValueError(f"Invalid target language: {target_lang}")

    if not self.encoder or not self.decoder:
        raise RuntimeError("Models not initialized")

    # --- MODIFIED ---
    # 1. Determine which vocabulary and adapter to use
    if domain:
        # Construct domain-specific names
        vocab_pack_name = f"latin_{domain}" # e.g., 'latin_medical'
        adapter_name = f"{source_lang}_{domain}" # e.g., 'es_medical'
    else:
        # Fallback to general-purpose packs
        vocab_pack_name = self.vocab_manager.language_to_pack.get(source_lang, 'latin')
        adapter_name = source_lang

    # 2. Load the correct vocabulary pack
    try:
        vocab_pack = self.vocab_manager.get_vocab_for_pair(source_lang, target_lang)
    except Exception:
        if domain:
            logger.warning(f"Domain vocab '{vocab_pack_name}' not found. Falling back to general vocab.")
            general_pack_name = self.vocab_manager.language_to_pack.get(source_lang, 'latin')
            vocab_pack = self.vocab_manager.get_vocab_for_pair(source_lang, target_lang)
            # Also fallback the adapter name
            adapter_name = source_lang
        else:
            raise # Re-raise if general vocab is not found

    # 3. Load the correct adapter and translate (atomic under lock)
    with self._model_lock:
        self.encoder.load_language_adapter(adapter_name, adapter_path=f"{MODELS_ADAPTERS_DIR}/best_{adapter_name}_adapter.pt")

        # 4. Translate
        if self.evaluator:
            return self.evaluator.translate(text, source_lang, target_lang)
        else:
            raise RuntimeError("Translation system not fully initialized")


async def translate_async(self, text: str, source_lang: str, target_lang: str) -> str:
    """Async translation for better concurrency"""
    # Record metrics
    start_time = time.time()
    translation_counter.labels(source_lang=source_lang, target_lang=target_lang).inc()

    # Run CPU-bound translation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        self.executor,
        self.translate,
        text,
        source_lang,
        target_lang
    )

    # Record duration
    translation_duration.observe(time.time() - start_time)

    return result


async def translate_batch_async(self,
                               texts: List[str],
                               source_lang: str,
                               target_lang: str,
                               max_concurrent: int = 10) -> List[str]:
    """Translate multiple texts concurrently with rate limiting"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_with_semaphore(text: str) -> str:
        async with semaphore:
            return await self.translate_async(text, source_lang, target_lang)

    tasks = [translate_with_semaphore(text) for text in texts]
    return await asyncio.gather(*tasks)


def evaluate(self, test_file: str, output_file: Optional[str] = None):
    """Evaluate the system on test data"""
    if not self.evaluator:
        logger.error("❌ Evaluation system not initialized")
        return

    logger.info(f"📊 Evaluating on {test_file}...")

    metrics = self.evaluator.evaluate_file(test_file)

    if output_file:
        self.evaluator.create_evaluation_report(metrics, output_file)

    return metrics


async def evaluate_async(self, validation_data: str) -> Dict[str, float]:
    """Async evaluation wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        self.executor,
        self.evaluate,
        validation_data
    )


# Patch methods onto UniversalTranslationSystem
UniversalTranslationSystem.translate = translate
UniversalTranslationSystem.translate_async = translate_async
UniversalTranslationSystem.translate_batch_async = translate_batch_async
UniversalTranslationSystem.evaluate = evaluate
UniversalTranslationSystem.evaluate_async = evaluate_async


# --- Integration pipeline functions ---

async def integrate_full_pipeline_async(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Main async integration function that loads config from a file."""
    config_path = config_file or f"{CONFIG_DIR}/deployment_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config = SystemConfig(**config_data)
        logger.info(f"Loaded deployment configuration from: {config_path}")
    except (FileNotFoundError, yaml.YAMLError, TypeError) as e:
        logger.error(f"Failed to load deployment config '{config_path}': {e}. Using default settings.")
        config = SystemConfig()

    # Create the system with the validated config object
    system = UniversalTranslationSystem(config)

    # Initialize all components
    if system.initialize_all_systems():
        logger.info("\n✅ System ready for use!")
        health = await system.health_check_async()
        logger.info(f"System health: {health['status']}")
        return system

    logger.error("❌ System initialization failed")
    return None


# Synchronous wrapper for backward compatibility
def integrate_full_pipeline(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Synchronous wrapper for the async integration"""
    return asyncio.run(integrate_full_pipeline_async(config_file))
