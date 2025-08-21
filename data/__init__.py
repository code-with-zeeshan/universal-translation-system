# data/__init__.py
"""
Data pipeline module for the Universal Translation System.

This module handles data downloading, processing, sampling, and augmentation
for training multilingual translation models.
"""

from .data_utils import ConfigManager, DataProcessor, DatasetLoader
from .unified_data_downloader import UnifiedDataDownloader

# Backward compatibility aliases
CuratedDataDownloader = UnifiedDataDownloader
MultilingualDataCollector = UnifiedDataDownloader
SmartDataStrategy = UnifiedDataDownloader
from .smart_sampler import SmartDataSampler
from .synthetic_augmentation import SyntheticDataAugmenter
from .pipeline_connector import PipelineConnector
from .unified_data_pipeline import UnifiedDataPipeline as PracticalDataPipeline

__all__ = [
    "ConfigManager",
    "DataProcessor",
    "DatasetLoader",
    "CuratedDataDownloader",
    "MultilingualDataCollector",
    "SmartDataStrategy",
    "SmartDataSampler",
    "SyntheticDataAugmenter",
    "PipelineConnector",
    "PracticalDataPipeline"
]