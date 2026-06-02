from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import yaml
from pathlib import Path
import logging
import os
import json
from utils.constants import MODELS_PRODUCTION_DIR, ENCODER_MODEL_FILENAME, VOCAB_DIR, DECODER_MODEL_FILENAME

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Schema for the 'data' section of the config."""
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    training_distribution: Dict[str, int]
    active_languages: Optional[List[str]] = None
    quality_threshold: float = Field(0.8, description="Quality threshold for data filtering", ge=0.0, le=1.0)
    total_size_gb: float = Field(8.0, description="Target total dataset size in GB")
    max_sentence_length: int = Field(64, description="Maximum sentence length")
    output_dir: str = "data/processed"
    augmentation_pairs: List[str] = Field(default_factory=list, description="Language pairs for synthetic augmentation")

    class Config:
        extra = "allow"


class ModelConfig(BaseModel):
    """Schema for the 'model' section."""
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    decoder_dim: int = 512
    decoder_layers: int = 6
    decoder_heads: int = 8
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    max_vocab_size: int = 32000

    class Config:
        extra = "allow"


class MemoryConfig(BaseModel):
    """Schema for memory optimization settings."""
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    flash_attention: bool = True
    dtype: str = "bfloat16"
    use_channels_last: bool = False
    max_split_size: int = 512
    empty_cache_freq: int = 100

    class Config:
        extra = "allow"


class VocabularyConfig(BaseModel):
    """Schema for vocabulary configuration."""
    language_to_pack_mapping: Dict[str, str] = {
        "en": "latin", "es": "latin", "fr": "latin", "de": "latin",
        "it": "latin", "pt": "latin", "nl": "latin", "sv": "latin",
        "pl": "latin", "id": "latin", "vi": "latin", "tr": "latin",
        "zh": "cjk", "ja": "cjk", "ko": "cjk",
        "ar": "arabic", "hi": "devanagari",
        "ru": "cyrillic", "uk": "cyrillic", "th": "thai"
    }
    vocab_dir: str = "vocabulary/vocab"
    dynamic_vocabulary: bool = True
    vocab_switch_penalty: float = 0.001

    class Config:
        extra = "allow"


class TrainingConfig(BaseModel):
    """Configuration for training. Maps from base.yaml 'training' section."""
    use_fsdp: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    flash_attention: bool = True
    accumulation_steps: int = Field(4, gt=0)
    max_grad_norm: float = 1.0
    lr: float = Field(5e-5, gt=0)
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_every: int = 1000
    log_every: int = 100
    validate_every: int = 1
    profile_training: bool = False
    num_epochs: Optional[int] = 20
    batch_size: int = 32
    resume_from: Optional[str] = None
    validate_only: Optional[bool] = False
    checkpoint: Optional[str] = None
    dynamic_vocabulary: bool = True
    vocab_switch_penalty: float = 0.001
    language_to_pack_mapping: Optional[Dict[str, str]] = None
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_rslora: bool = True

    class Config:
        extra = "allow"


class MonitoringConfig(BaseModel):
    """Configuration for monitoring, logging, and observability."""
    use_wandb: bool = False
    log_every: int = 10
    save_every: int = 1000
    profile: bool = False
    prometheus_port: int = Field(
        default=int(os.environ.get("PROMETHEUS_PORT", "9090")),
        description="Port for Prometheus"
    )
    grafana_port: int = Field(
        default=int(os.environ.get("GRAFANA_PORT", "3000")),
        description="Port for Grafana"
    )
    metrics_path: str = Field(
        default=os.environ.get("METRICS_PATH", "/metrics"),
        description="Path for metrics endpoint"
    )
    log_level: str = Field(
        default=os.environ.get("MONITORING_LOG_LEVEL", "INFO"),
        description="Log level for monitoring"
    )
    collection_interval: int = Field(
        default=int(os.environ.get("METRICS_COLLECTION_INTERVAL", "15")),
        description="Interval for metrics collection in seconds"
    )
    enable_system_metrics: bool = Field(
        default=os.environ.get("ENABLE_SYSTEM_METRICS", "true").lower() in ["true", "1", "yes"],
        description="Enable system metrics collection"
    )
    enable_gpu_metrics: bool = Field(
        default=os.environ.get("ENABLE_GPU_METRICS", "true").lower() in ["true", "1", "yes"],
        description="Enable GPU metrics collection"
    )
    enable_vocabulary_metrics: bool = Field(
        default=os.environ.get("ENABLE_VOCABULARY_METRICS", "true").lower() in ["true", "1", "yes"],
        description="Enable vocabulary metrics collection"
    )

    class Config:
        extra = "allow"


class PipelineConfig(BaseModel):
    """Schema for the 'pipeline' section."""
    enabled_stages: List[str] = Field(
        default_factory=lambda: [
            "download_evaluation", "download_training", "sample_filter",
            "augment", "create_ready", "validate", "vocabulary"
        ]
    )
    comet_quality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_dynamic_ff_per_pair: int = 5000
    max_idiom_per_lang: int = 2000

    class Config:
        extra = "allow"


class RootConfig(BaseModel):
    """The root configuration model loaded from base.yaml for the training pipeline."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    memory: MemoryConfig
    vocabulary: VocabularyConfig
    monitoring: Optional[MonitoringConfig] = None
    pipeline: Optional[PipelineConfig] = None
    tier_metadata: Optional[dict] = None

    class Config:
        extra = "allow"


class EncoderConfig(BaseModel):
    """Configuration for the encoder component. Supports env-var overrides."""
    model_path: str = Field(
        default=os.environ.get("ENCODER_MODEL_PATH", f"{MODELS_PRODUCTION_DIR}/{ENCODER_MODEL_FILENAME}"),
        description="Path to encoder model"
    )
    embedding_dim: int = Field(
        default=int(os.environ.get("EMBEDDING_DIM", "768")),
        description="Dimension of embeddings"
    )
    vocab_dir: str = Field(
        default=os.environ.get("VOCAB_DIR", VOCAB_DIR),
        description="Directory containing vocabulary files"
    )
    max_sequence_length: int = Field(
        default=int(os.environ.get("MAX_SEQUENCE_LENGTH", "512")),
        description="Maximum sequence length for encoding"
    )
    device: str = Field(
        default=os.environ.get("ENCODER_DEVICE", "auto"),
        description="Device to run encoder on (auto, cpu, cuda)"
    )
    fallback_model_path: Optional[str] = Field(
        default=os.environ.get("FALLBACK_MODEL_PATH", None),
        description="Path to fallback model"
    )

    @validator("device")
    def validate_device(cls, v):
        if v not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Device must be one of: auto, cpu, cuda. Got: {v}")
        return v


class DecoderConfig(BaseModel):
    """Configuration for the decoder component. Supports env-var overrides."""
    model_path: str = Field(
        default=os.environ.get("DECODER_MODEL_PATH", f"{MODELS_PRODUCTION_DIR}/{DECODER_MODEL_FILENAME}"),
        description="Path to decoder model"
    )
    vocab_dir: str = Field(
        default=os.environ.get("VOCAB_DIR", VOCAB_DIR),
        description="Directory containing vocabulary files"
    )
    max_sequence_length: int = Field(
        default=int(os.environ.get("MAX_SEQUENCE_LENGTH", "512")),
        description="Maximum sequence length for decoding"
    )
    device: str = Field(
        default=os.environ.get("DECODER_DEVICE", "cuda"),
        description="Device to run decoder on (cpu, cuda)"
    )
    batch_size: int = Field(
        default=int(os.environ.get("DECODER_BATCH_SIZE", "32")),
        description="Batch size for decoding"
    )
    beam_size: int = Field(
        default=int(os.environ.get("DECODER_BEAM_SIZE", "5")),
        description="Beam size for decoding"
    )
    max_batch_tokens: int = Field(
        default=int(os.environ.get("MAX_BATCH_TOKENS", "8192")),
        description="Maximum number of tokens in a batch"
    )

    @validator("device")
    def validate_device(cls, v):
        if v not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be one of: cpu, cuda. Got: {v}")
        return v


class CoordinatorConfig(BaseModel):
    """Configuration for the coordinator component. Supports env-var overrides."""
    host: str = Field(
        default=os.environ.get("COORDINATOR_HOST", "0.0.0.0"),
        description="Host to bind coordinator to"
    )
    port: int = Field(
        default=int(os.environ.get("COORDINATOR_PORT", "8002")),
        description="Port to bind coordinator to"
    )
    workers: int = Field(
        default=int(os.environ.get("COORDINATOR_WORKERS", "1")),
        description="Number of worker processes"
    )
    title: str = Field(
        default=os.environ.get("COORDINATOR_TITLE", "Universal Translation Coordinator"),
        description="Title for the coordinator API"
    )
    decoder_pool: List[str] = Field(
        default=os.environ.get("DECODER_POOL", "decoder:8001").split(","),
        description="Comma-separated list of decoder endpoints"
    )
    jwt_secret: str = Field(
        default=os.environ.get("COORDINATOR_JWT_SECRET", ""),
        description="Secret for JWT authentication"
    )
    token_expiry: int = Field(
        default=int(os.environ.get("TOKEN_EXPIRY", "3600")),
        description="JWT token expiry time in seconds"
    )

    @validator("jwt_secret")
    def validate_jwt_secret(cls, v):
        if not v:
            raise ValueError("JWT secret must be set")
        if len(v) < 32:
            raise ValueError("JWT secret should be at least 32 characters long")
        return v


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker. Supports env-var overrides."""
    failure_threshold: int = Field(
        default=int(os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
        description="Number of failures before opening circuit"
    )
    recovery_timeout: int = Field(
        default=int(os.environ.get("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30")),
        description="Seconds to wait before testing circuit again"
    )
    timeout: int = Field(
        default=int(os.environ.get("CIRCUIT_BREAKER_TIMEOUT", "10")),
        description="Request timeout in seconds"
    )


class SystemConfig(BaseModel):
    """Complete system configuration for the serving stack. Supports env-var overrides and file loading."""
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "SystemConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "SystemConfig":
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        with open(file_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

    def to_json(self, file_path: Union[str, Path]) -> None:
        with open(file_path, "w") as f:
            json.dump(self.dict(), f, indent=2)


def load_config(config_path: str = "config/base.yaml", base_config: Optional[RootConfig] = None) -> RootConfig:
    """
    Load and validate configuration from YAML file.
    Returns a RootConfig (training-pipeline oriented).
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return RootConfig(
            data=DataConfig(training_distribution={}),
            model=ModelConfig(),
            training=TrainingConfig(),
            memory=MemoryConfig(),
            vocabulary=VocabularyConfig()
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if 'data' not in config_data:
            config_data['data'] = {'training_distribution': {}}

        if 'model' not in config_data:
            config_data['model'] = {}

        if 'training' not in config_data:
            config_data['training'] = {}

        if 'memory' not in config_data:
            memory_settings = {}
            training_data = config_data.get('training', {})

            memory_fields = [
                'mixed_precision', 'gradient_checkpointing', 'cpu_offload',
                'activation_checkpointing', 'compile_model', 'compile_mode',
                'flash_attention', 'dtype', 'use_channels_last', 'max_split_size',
                'empty_cache_freq'
            ]

            for field in memory_fields:
                if field in training_data:
                    memory_settings[field] = training_data[field]

            config_data['memory'] = memory_settings

        if 'vocabulary' not in config_data:
            vocab_settings = {}
            training_data = config_data.get('training', {})

            if 'language_to_pack_mapping' in training_data:
                vocab_settings['language_to_pack_mapping'] = training_data['language_to_pack_mapping']
            if 'dynamic_vocabulary' in training_data:
                vocab_settings['dynamic_vocabulary'] = training_data['dynamic_vocabulary']
            if 'vocab_switch_penalty' in training_data:
                vocab_settings['vocab_switch_penalty'] = training_data['vocab_switch_penalty']

            config_data['vocabulary'] = vocab_settings

        if 'active_languages' not in config_data['data']:
            training_dist = config_data['data'].get('training_distribution', {})
            languages = set()
            for pair in training_dist.keys():
                if '-' in pair:
                    src, tgt = pair.split('-', 1)
                    languages.update([src, tgt])
            config_data['data']['active_languages'] = list(languages)

        config = RootConfig(**config_data)

        if base_config:
            merged_data = base_config.dict()
            loaded_data = config.dict()

            def deep_merge(base_dict, override_dict):
                for key, value in override_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_merge(base_dict[key], value)
                    else:
                        base_dict[key] = value

            deep_merge(merged_data, loaded_data)
            config = RootConfig(**merged_data)

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def load_pydantic_config(config_path: str = "config/base.yaml", base_config: Optional[RootConfig] = None) -> RootConfig:
    """Alias for load_config for backward compatibility."""
    return load_config(config_path, base_config)


def load_system_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """
    Load system configuration from file (YAML/JSON) or environment variables.
    Returns a SystemConfig (serving-stack oriented).
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if path.suffix.lower() in [".yaml", ".yml"]:
            return SystemConfig.from_yaml(path)
        elif path.suffix.lower() == ".json":
            return SystemConfig.from_json(path)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    return SystemConfig()
