# training/training_validator.py
"""
Validation utilities for training configurations and models
"""
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TrainingValidator:
    """Validate training configurations and models"""
    
    @staticmethod
    def validate_training_config(config: Any) -> List[str]:
        """Validate training configuration"""
        errors = []
        
        # Check required fields
        required_fields = ['lr', 'batch_size', 'num_epochs']
        for field in required_fields:
            if not hasattr(config, field):
                errors.append(f"Missing required field: {field}")
        
        # Validate ranges
        if hasattr(config, 'lr'):
            if not 0 < config.lr < 1:
                errors.append(f"Invalid learning rate: {config.lr}")
        
        if hasattr(config, 'batch_size'):
            if not 1 <= config.batch_size <= 1024:
                errors.append(f"Invalid batch size: {config.batch_size}")
        
        # Validate device settings
        if hasattr(config, 'device'):
            if config.device == 'cuda' and not torch.cuda.is_available():
                errors.append("CUDA requested but not available")
        
        # Validate distributed settings
        if hasattr(config, 'use_fsdp') and config.use_fsdp:
            if not torch.distributed.is_available():
                errors.append("FSDP requested but distributed not available")
        
        return errors
    
    @staticmethod
    def validate_model_compatibility(encoder: torch.nn.Module, 
                                   decoder: torch.nn.Module) -> List[str]:
        """Validate encoder-decoder compatibility"""
        errors = []
        
        # Check dimensions match
        if hasattr(encoder, 'hidden_dim') and hasattr(decoder, 'encoder_dim'):
            if encoder.hidden_dim != decoder.encoder_dim:
                errors.append(
                    f"Dimension mismatch: encoder hidden_dim={encoder.hidden_dim}, "
                    f"decoder encoder_dim={decoder.encoder_dim}"
                )
        
        # Check both models are on same device
        encoder_device = next(encoder.parameters()).device
        decoder_device = next(decoder.parameters()).device
        
        if encoder_device != decoder_device:
            errors.append(
                f"Device mismatch: encoder on {encoder_device}, "
                f"decoder on {decoder_device}"
            )
        
        return errors
    
    @staticmethod
    def validate_checkpoint(checkpoint_path: str, 
                          expected_keys: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Validate checkpoint file"""
        errors = []
        
        if not Path(checkpoint_path).exists():
            return False, ["Checkpoint file does not exist"]
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check expected keys
            if expected_keys is None:
                expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            
            for key in expected_keys:
                if key not in checkpoint:
                    errors.append(f"Missing key in checkpoint: {key}")
            
            # Check for corruption
            if 'model_state_dict' in checkpoint:
                for key, tensor in checkpoint['model_state_dict'].items():
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        errors.append(f"Invalid values in checkpoint tensor: {key}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Failed to load checkpoint: {str(e)}"]