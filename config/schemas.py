# config/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class DataConfig(BaseModel):
    """Schema for the 'data' section of the config."""
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    training_distribution: Dict[str, int]
    active_languages: Optional[List[str]] = None
    quality_threshold: float = Field(0.8, description="Quality threshold for data filtering", ge=0.0, le=1.0)

class ModelConfig(BaseModel):
    """Schema for the 'model' section."""
    vocab_size: int = 50000
    hidden_dim: int = 1024
    num_layers: int = 6
    num_heads: int = 16
    decoder_dim: int = 512
    decoder_layers: int = 6
    decoder_heads: int = 8
    dropout: float = Field(0.1, ge=0.0, le=1.0)

class TrainingConfig(BaseModel):
    """Schema for the 'training' section."""
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
    profile_training: bool = False
    num_epochs: Optional[int] = 20
    resume_from: Optional[str] = None
    validate_only: Optional[bool] = False
    checkpoint: Optional[str] = None

class RootConfig(BaseModel):
    """The root configuration model."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tier_metadata: Optional[dict] = None