# config/__init__.py
"""
Config package for models and schema loading utilities.

Note: Import from config.config_models for the consolidated SystemConfig and
helpers, or from config.schemas for schema-based loading where needed.
"""
# Lazy import wrappers to avoid hard dependency on specific Pydantic major versions
# Import config_models only when explicitly requested, so schemas-based loader can be used independently.

def _lazy_import_config_models():
    from .config_models import (
        EncoderConfig,
        DecoderConfig,
        CoordinatorConfig,
        CircuitBreakerConfig,
        MonitoringConfig,
        TrainingConfig,
        load_config as load_system_config,
    )
    return {
        "EncoderConfig": EncoderConfig,
        "DecoderConfig": DecoderConfig,
        "CoordinatorConfig": CoordinatorConfig,
        "CircuitBreakerConfig": CircuitBreakerConfig,
        "MonitoringConfig": MonitoringConfig,
        "TrainingConfig": TrainingConfig,
        "load_system_config": load_system_config,
    }

# Expose schema-based loader directly (v2-friendly)
try:
    from .schemas import load_config as load_yaml_config  # type: ignore
    from .schemas import RootConfig  # type: ignore
except Exception:  # pragma: no cover
    load_yaml_config = None  # type: ignore
    RootConfig = None  # type: ignore

# Support attribute access for config_models symbols but import lazily
def __getattr__(name):  # type: ignore
    models = _lazy_import_config_models()
    if name in models:
        return models[name]
    raise AttributeError(name)

__all__ = [
    # Lazy-exported config models
    "EncoderConfig",
    "DecoderConfig",
    "CoordinatorConfig",
    "CircuitBreakerConfig",
    "MonitoringConfig",
    "TrainingConfig",
    "load_system_config",
    # Schemas loader
    "load_yaml_config",
    "RootConfig",
]