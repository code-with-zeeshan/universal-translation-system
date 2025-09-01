"""
Configuration models for the Universal Translation System.
Uses Pydantic for validation and environment variable support.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator
import yaml
import json

# Optional dependency: python-dotenv
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

# Load environment variables from .env file if present (no-op if dotenv missing)
try:
    load_dotenv()
except Exception:
    pass


class EncoderConfig(BaseModel):
    """Configuration for the encoder component."""
    model_path: str = Field(
        default=os.environ.get("ENCODER_MODEL_PATH", "models/production/encoder.pt"),
        description="Path to encoder model"
    )
    embedding_dim: int = Field(
        default=int(os.environ.get("EMBEDDING_DIM", "768")),
        description="Dimension of embeddings"
    )
    vocab_dir: str = Field(
        default=os.environ.get("VOCAB_DIR", "vocabs"),
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
        """Validate device setting."""
        if v not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Device must be one of: auto, cpu, cuda. Got: {v}")
        return v
    
    @root_validator
    def check_model_path(cls, values):
        """Check that model path exists."""
        model_path = values.get("model_path")
        if model_path and not Path(model_path).exists():
            # Just warn, don't fail - model might be downloaded later
            print(f"Warning: Model path does not exist: {model_path}")
        return values


class DecoderConfig(BaseModel):
    """Configuration for the decoder component."""
    model_path: str = Field(
        default=os.environ.get("DECODER_MODEL_PATH", "models/production/decoder.pt"),
        description="Path to decoder model"
    )
    vocab_dir: str = Field(
        default=os.environ.get("VOCAB_DIR", "vocabs"),
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
        """Validate device setting."""
        if v not in ["cpu", "cuda"]:
            raise ValueError(f"Device must be one of: cpu, cuda. Got: {v}")
        return v


class CoordinatorConfig(BaseModel):
    """Configuration for the coordinator component."""
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
        """Validate JWT secret."""
        if not v:
            raise ValueError("JWT secret must be set")
        if len(v) < 32:
            raise ValueError("JWT secret should be at least 32 characters long")
        return v


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
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


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""
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


class TrainingConfig(BaseModel):
    """Configuration for training."""
    batch_size: int = Field(
        default=int(os.environ.get("TRAINING_BATCH_SIZE", "32")),
        description="Batch size for training"
    )
    epochs: int = Field(
        default=int(os.environ.get("TRAINING_EPOCHS", "20")),
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=float(os.environ.get("TRAINING_LEARNING_RATE", "5e-5")),
        description="Learning rate for training"
    )
    weight_decay: float = Field(
        default=float(os.environ.get("TRAINING_WEIGHT_DECAY", "0.01")),
        description="Weight decay for training"
    )
    warmup_steps: int = Field(
        default=int(os.environ.get("TRAINING_WARMUP_STEPS", "1000")),
        description="Number of warmup steps"
    )
    gradient_accumulation_steps: int = Field(
        default=int(os.environ.get("TRAINING_GRADIENT_ACCUMULATION_STEPS", "1")),
        description="Gradient accumulation steps"
    )
    mixed_precision: bool = Field(
        default=os.environ.get("TRAINING_MIXED_PRECISION", "true").lower() in ["true", "1", "yes"],
        description="Enable mixed precision training"
    )
    gradient_checkpointing: bool = Field(
        default=os.environ.get("TRAINING_GRADIENT_CHECKPOINTING", "false").lower() in ["true", "1", "yes"],
        description="Enable gradient checkpointing"
    )
    log_steps: int = Field(
        default=int(os.environ.get("TRAINING_LOG_STEPS", "100")),
        description="Log interval in steps"
    )
    save_steps: int = Field(
        default=int(os.environ.get("TRAINING_SAVE_STEPS", "1000")),
        description="Save interval in steps"
    )
    eval_steps: int = Field(
        default=int(os.environ.get("TRAINING_EVAL_STEPS", "1000")),
        description="Evaluation interval in steps"
    )


class SystemConfig(BaseModel):
    """Complete system configuration."""
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.parse_obj(config_dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "SystemConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls.parse_obj(config_dict)
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.dict(), f, indent=2)


def load_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """
    Load system configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        SystemConfig: Complete system configuration
    """
    # If config path is provided, load from file
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
    
    # Otherwise, create from environment variables
    return SystemConfig()