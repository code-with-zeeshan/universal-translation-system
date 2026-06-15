# integration/translation_api.py
"""
Translation API with async support for the Universal Translation System
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import yaml
from utils.common_utils import RuntimeDirectoryManager

from .system import UniversalTranslationSystem
from .system_config import SystemConfig

logger = logging.getLogger(__name__)


# --- Integration pipeline functions ---

async def integrate_full_pipeline_async(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Main async integration function that loads config from a file."""
    config_path = config_file or str(RuntimeDirectoryManager().generated_config_dir / "deployment_config.yaml")
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
