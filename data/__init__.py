# data/__init__.py
"""
Data pipeline module for the Universal Translation System.

Handles data downloading, processing, sampling, and augmentation.
"""

# Avoid importing heavy optional deps at package import time in smoke/dry-run
from .data_utils import DataProcessor, DatasetLoader

# Optional: downloader may require 'requests'. Defer failure until actually used.
try:
    from .unified_data_downloader import UnifiedDataDownloader  # type: ignore
except Exception:
    UnifiedDataDownloader = None  # type: ignore

from .smart_sampler import SmartDataSampler
from .synthetic_augmentation import SyntheticDataAugmenter
from .unified_data_pipeline import UnifiedDataPipeline as PracticalDataPipeline

# Cross-package dependency: PipelineConnector lives in connector package
try:
    from connector.pipeline_connector import PipelineConnector  # type: ignore
except Exception:
    PipelineConnector = None  # type: ignore

# Backward compatibility aliases
CuratedDataDownloader = UnifiedDataDownloader
MultilingualDataCollector = UnifiedDataDownloader
SmartDataStrategy = UnifiedDataDownloader

__all__ = [
    "DataProcessor",
    "DatasetLoader",
    "CuratedDataDownloader",
    "MultilingualDataCollector",
    "SmartDataStrategy",
    "SmartDataSampler",
    "SyntheticDataAugmenter",
    "PracticalDataPipeline",
]
if PipelineConnector is not None:
    __all__.append("PipelineConnector")