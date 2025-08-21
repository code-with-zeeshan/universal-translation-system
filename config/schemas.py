# config/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataConfig(BaseModel):
    """Schema for the 'data' section of the config."""
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    training_distribution: Dict[str, int]
    active_languages: Optional[List[str]] = None
    quality_threshold: float = Field(0.8, description="Quality threshold for data filtering", ge=0.0, le=1.0)
    total_size_gb: float = Field(8.0, description="Target total dataset size in GB")
    max_sentence_length: int = Field(50, description="Maximum sentence length")
    output_dir: str = "data/processed"

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
    max_vocab_size: int = 50000

class MemoryConfig(BaseModel):
    """Schema for memory optimization settings."""
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    flash_attention: bool = True
    use_flash_attention: bool = True
    dtype: str = "bfloat16"
    use_channels_last: bool = False
    max_split_size: int = 512
    empty_cache_freq: int = 100

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
    vocab_dir: str = "vocabs"
    dynamic_vocabulary: bool = True
    vocab_switch_penalty: float = 0.001

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
    learning_rate: float = Field(5e-4, gt=0)  # Alternative name
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

class RootConfig(BaseModel):
    """The root configuration model."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    memory: MemoryConfig
    vocabulary: VocabularyConfig
    tier_metadata: Optional[dict] = None

    class Config:
        extra = "allow"  # Allow extra fields for flexibility

def load_config(config_path: str = "config/base.yaml", base_config: Optional[RootConfig] = None) -> RootConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        base_config: Optional base configuration to merge with
        
    Returns:
        Validated RootConfig instance
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
        
        # Handle missing sections with defaults
        if 'data' not in config_data:
            config_data['data'] = {'training_distribution': {}}
        
        if 'model' not in config_data:
            config_data['model'] = {}
            
        if 'training' not in config_data:
            config_data['training'] = {}
            
        if 'memory' not in config_data:
            # Extract memory settings from training if they exist
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
            # Extract vocabulary settings from training if they exist
            vocab_settings = {}
            training_data = config_data.get('training', {})
            
            if 'language_to_pack_mapping' in training_data:
                vocab_settings['language_to_pack_mapping'] = training_data['language_to_pack_mapping']
            if 'dynamic_vocabulary' in training_data:
                vocab_settings['dynamic_vocabulary'] = training_data['dynamic_vocabulary']
            if 'vocab_switch_penalty' in training_data:
                vocab_settings['vocab_switch_penalty'] = training_data['vocab_switch_penalty']
                
            config_data['vocabulary'] = vocab_settings
        
        # Ensure active_languages is set
        if 'active_languages' not in config_data['data']:
            # Extract from training_distribution keys
            training_dist = config_data['data'].get('training_distribution', {})
            languages = set()
            for pair in training_dist.keys():
                if '-' in pair:
                    src, tgt = pair.split('-', 1)
                    languages.update([src, tgt])
            config_data['data']['active_languages'] = list(languages)
        
        # Create and validate config
        config = RootConfig(**config_data)
        
        # Merge with base config if provided
        if base_config:
            # Simple merge - override base with loaded values
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