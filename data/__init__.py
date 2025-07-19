# data/__init__.py
"""
Data pipeline module for the Universal Translation System.

This module handles data downloading, processing, sampling, and augmentation
for training multilingual translation models.
"""

from .data_utils import ConfigManager, DataProcessor, DatasetLoader
from .download_curated_data import CuratedDataDownloader
from .download_training_data import MultilingualDataCollector
from .smart_data_downloader import SmartDataStrategy
from .smart_sampler import SmartDataSampler
from .synthetic_augmentation import SyntheticDataAugmenter
from .pipeline_connector import PipelineConnector
from .practical_data_pipeline import PracticalDataPipeline

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