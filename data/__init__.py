# data/__init__.py
"""
Data pipeline module for the Universal Translation System.

Handles data downloading, processing, sampling, and augmentation.
"""

from .data_utils import DataProcessor, DatasetLoader
from .unified_data_downloader import UnifiedDataDownloader
from .smart_sampler import SmartDataSampler
from .synthetic_augmentation import SyntheticDataAugmenter
from .unified_data_pipeline import UnifiedDataPipeline as PracticalDataPipeline

# Cross-package dependency: PipelineConnector lives in connector package
from connector.pipeline_connector import PipelineConnector

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
    "PipelineConnector",
    "PracticalDataPipeline",
]