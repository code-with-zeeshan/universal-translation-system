# config/__init__.py
"""
Config package for models and schema loading utilities.

Note: Import from config.config_models for the consolidated SystemConfig and
helpers, or from config.schemas for schema-based loading where needed.
"""
from .config_models import (
    EncoderConfig,
    DecoderConfig,
    CoordinatorConfig,
    CircuitBreakerConfig,
    MonitoringConfig,
    TrainingConfig,
    load_config as load_system_config,
)

try:
    # Optional: schema-based loader if present
    from .schemas import load_config as load_yaml_config  # type: ignore
except Exception:  # pragma: no cover - schemas may be optional in some envs
    load_yaml_config = None  # type: ignore

__all__ = [
    "EncoderConfig",
    "DecoderConfig",
    "CoordinatorConfig",
    "CircuitBreakerConfig",
    "MonitoringConfig",
    "TrainingConfig",
    "load_system_config",
    "load_yaml_config",
]