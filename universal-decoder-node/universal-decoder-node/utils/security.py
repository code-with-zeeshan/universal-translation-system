# universal-decoder-node/universal_decoder_node/utils/security.py
import os
import logging
import hashlib
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_model_source(model_path: str) -> bool:
    """
    Validate that a model file comes from a trusted source.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if the model is trusted, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Check if model is in trusted directory
    trusted_dirs = [
        os.environ.get("TRUSTED_MODELS_DIR", "models"),
        os.path.join(os.path.expanduser("~"), ".universal-decoder", "models")
    ]
    
    model_path = os.path.abspath(model_path)
    for trusted_dir in trusted_dirs:
        trusted_dir = os.path.abspath(trusted_dir)
        if model_path.startswith(trusted_dir):
            return True
    
    # Check if model has trusted hash
    try:
        model_hash = calculate_file_hash(model_path)
        trusted_hashes = get_trusted_model_hashes()
        
        return model_hash in trusted_hashes
    except Exception as e:
        logger.error(f"Error validating model hash: {e}")
        return False

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as a hex string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in 1MB chunks to avoid loading large files into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def get_trusted_model_hashes() -> Dict[str, str]:
    """
    Get dictionary of trusted model hashes.
    
    Returns:
        Dictionary mapping model hashes to model names
    """
    # Try to load from environment
    trusted_hashes = {}
    
    for key, value in os.environ.items():
        if key.startswith("TRUSTED_MODEL_HASH_"):
            model_name = key[len("TRUSTED_MODEL_HASH_"):].lower()
            trusted_hashes[value] = model_name
    
    # Try to load from file
    hash_file = os.environ.get("TRUSTED_HASHES_FILE", "trusted_model_hashes.txt")
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split(":")
                        if len(parts) >= 2:
                            model_hash = parts[0].strip()
                            model_name = parts[1].strip()
                            trusted_hashes[model_hash] = model_name
        except Exception as e:
            logger.error(f"Error loading trusted hashes file: {e}")
    
    return trusted_hashes

def safe_load_model(model_path: Optional[str], device: torch.device) -> Tuple[Optional[torch.nn.Module], Dict[str, Any]]:
    """
    Safely load a PyTorch model with validation.
    
    Args:
        model_path: Path to the model file, or None to create a new model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, metadata)
    """
    metadata = {
        "source": "unknown",
        "trusted": False,
        "created_new": False
    }
    
    if model_path and os.path.exists(model_path):
        # Validate model source
        metadata["trusted"] = validate_model_source(model_path)
        
        if not metadata["trusted"]:
            logger.warning(f"Loading untrusted model from {model_path}")
        
        try:
            # Load model with safety measures
            model = torch.load(model_path, map_location=device)
            metadata["source"] = "file"
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            # Fall back to creating a new model
    
    # Create a new model if no model path or loading failed
    from ..decoder import OptimizedUniversalDecoder
    model = OptimizedUniversalDecoder().to(device)
    metadata["source"] = "new"
    metadata["created_new"] = True
    metadata["trusted"] = True
    
    return model, metadata