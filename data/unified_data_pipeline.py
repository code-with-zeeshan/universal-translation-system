# data/unified_data_pipeline.py
"""
Unified data pipeline manager combining all data management functionality.
Replaces: data_manager.py and practical_data_pipeline.py
"""

import logging
import os

from utils.logging_config import setup_logging
from data.pipeline_state import PipelineStage, PipelineState
from data.custom_samplers import BalancedLanguageSampler
from data.pipeline_orchestrator import UnifiedDataPipeline

# Ensure centralized logging for data pipeline
setup_logging(log_dir="logs", log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("data")


__all__ = [
    "PipelineStage",
    "PipelineState",
    "BalancedLanguageSampler",
    "UnifiedDataPipeline",
]


def main():
    """CLI entry point for the unified data pipeline"""
    from data.pipeline_orchestrator import main as orch_main
    orch_main()


if __name__ == "__main__":
    main()
