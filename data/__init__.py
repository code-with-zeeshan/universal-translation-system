# data/__init__.py
"""
Data pipeline module for the Universal Translation System.

Handles data downloading, processing, sampling, and augmentation.
"""

from .data_utils import DataProcessor, DatasetLoader
from .smart_sampler import SmartDataSampler

# Lightweight re-exports (no internal project imports)
__all__ = [
    "DataProcessor",
    "DatasetLoader",
    "UnifiedDataDownloader",
    "CuratedDataDownloader",
    "MultilingualDataCollector",
    "SmartDataStrategy",
    "SmartDataSampler",
    "SyntheticDataAugmenter",
    "PracticalDataPipeline",
]


def __getattr__(name):
    """Lazy-load heavy submodules on first attribute access.

    This avoids RuntimeWarning('found in sys.modules after import of package')
    when running via ``python -m data.unified_data_downloader``: the import
    chain triggered by ``__init__`` would otherwise add the target module to
    ``sys.modules`` *before* the ``-m`` runner can resolve it.
    """
    if name in ("UnifiedDataDownloader", "CuratedDataDownloader",
                 "MultilingualDataCollector", "SmartDataStrategy"):
        from data.unified_data_downloader import UnifiedDataDownloader as _cls
        for _alias in ("CuratedDataDownloader", "MultilingualDataCollector",
                       "SmartDataStrategy", "UnifiedDataDownloader"):
            globals()[_alias] = _cls
        return _cls

    if name == "SyntheticDataAugmenter":
        from data.synthetic_augmentation import SyntheticDataAugmenter as _cls
        globals()["SyntheticDataAugmenter"] = _cls
        return _cls

    if name == "PracticalDataPipeline":
        from data.unified_data_pipeline import UnifiedDataPipeline as _cls
        globals()["PracticalDataPipeline"] = _cls
        return _cls

    if name == "PipelineConnector":
        try:
            from connector.pipeline_connector import PipelineConnector as _cls
        except Exception:
            _cls = None
        globals()["PipelineConnector"] = _cls
        if _cls is not None:
            __all__.append("PipelineConnector")
        return _cls

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
