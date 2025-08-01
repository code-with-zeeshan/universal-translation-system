# utils/config_validator.py
"""
Configuration validation for the entire system
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validate system configuration files"""
    
    @staticmethod
    def validate_integration_config(config_path: str) -> List[str]:
        """Validate integration configuration"""
        errors = []
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return [f"Failed to load config: {e}"]
        
        # Required fields
        required = ['data_dir', 'model_dir', 'vocab_dir', 'device']
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate paths exist
        for dir_field in ['data_dir', 'model_dir', 'vocab_dir']:
            if dir_field in config:
                path = Path(config[dir_field])
                if not path.exists():
                    errors.append(f"{dir_field} does not exist: {path}")
        
        # Validate device
        if 'device' in config:
            valid_devices = ['cpu', 'cuda', 'mps', 'auto']
            if config['device'] not in valid_devices:
                errors.append(f"Invalid device: {config['device']}")
        
        # Validate numeric ranges
        if 'batch_size' in config:
            if not 1 <= config['batch_size'] <= 512:
                errors.append(f"Invalid batch_size: {config['batch_size']}")
        
        return errors
    
    @staticmethod
    def create_default_config(output_path: str = "config/default_config.yaml"):
        """Create default configuration file"""
        default_config = {
            'data_dir': 'data',
            'model_dir': 'models',
            'vocab_dir': 'vocabs',
            'checkpoint_dir': 'checkpoints',
            'device': 'auto',
            'use_adapters': True,
            'quantization_mode': 'fp32',
            'vocab_cache_size': 3,
            'batch_size': 32,
            'enable_monitoring': True,
            'monitoring_port': 8000,
            'security': {
                'validate_models': True,
                'allowed_model_sources': [
                    'facebook/',
                    'microsoft/',
                    'google/',
                    'Helsinki-NLP/'
                ]
            },
            'training': {
                'progressive': True,
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'compile_model': True
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created default configuration at {output_path}")
        return output_path