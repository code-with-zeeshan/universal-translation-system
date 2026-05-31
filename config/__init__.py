# config/__init__.py
"""
Config package for models and schema loading utilities.

All configuration models are now consolidated in config.schemas.
The old config.config_models module has been merged into config.schemas.
"""

from .schemas import (
    DataConfig,
    ModelConfig,
    MemoryConfig,
    VocabularyConfig,
    TrainingConfig,
    MonitoringConfig,
    RootConfig,
    EncoderConfig,
    DecoderConfig,
    CoordinatorConfig,
    CircuitBreakerConfig,
    SystemConfig,
    load_config,
    load_pydantic_config,
    load_system_config,
)

try:
    from .schemas import load_config as load_yaml_config  # type: ignore
except Exception:  # pragma: no cover
    load_yaml_config = None  # type: ignore

__all__ = [
    "DataConfig",
    "ModelConfig",
    "MemoryConfig",
    "VocabularyConfig",
    "TrainingConfig",
    "MonitoringConfig",
    "RootConfig",
    "EncoderConfig",
    "DecoderConfig",
    "CoordinatorConfig",
    "CircuitBreakerConfig",
    "SystemConfig",
    "load_config",
    "load_pydantic_config",
    "load_system_config",
    "load_yaml_config",
]