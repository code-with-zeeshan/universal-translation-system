#!/usr/bin/env python
# scripts/validate_config.py
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.schemas import load_config

def validate_config(config_path: str) -> List[str]:
    """
    Validate configuration file against schema
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation errors, empty if valid
    """
    try:
        config = load_config(config_path)
        # If we get here, validation passed
        return []
    except Exception as e:
        return [str(e)]

def check_config_references(config_path: str) -> List[str]:
    """
    Check if referenced files and directories exist
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of reference errors, empty if all valid
    """
    errors = []
    try:
        # Load the config file directly to check references
        # (We don't use load_config here because we want to check references even if schema validation fails)
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            return ["Unsupported config file format. Use .yaml, .yml, or .json"]
        
        # Check model paths
        if 'model_paths' in config_dict:
            for key, path in config_dict['model_paths'].items():
                if not Path(path).exists():
                    errors.append(f"Model path not found: {path} (referenced by {key})")
        
        # Check data paths
        if 'data_paths' in config_dict:
            for key, path in config_dict['data_paths'].items():
                if not Path(path).exists():
                    errors.append(f"Data path not found: {path} (referenced by {key})")
        
        # Check vocabulary paths
        if 'vocabulary' in config_dict and 'paths' in config_dict['vocabulary']:
            for key, path in config_dict['vocabulary']['paths'].items():
                if not Path(path).exists():
                    errors.append(f"Vocabulary path not found: {path} (referenced by {key})")
        
        # Check output paths
        if 'output_paths' in config_dict:
            for key, path in config_dict['output_paths'].items():
                parent_dir = Path(path).parent
                if not parent_dir.exists():
                    errors.append(f"Output directory not found: {parent_dir} (for {key})")
        
        return errors
    except Exception as e:
        return [f"Error checking config references: {e}"]

def check_config_consistency(config_path: str) -> List[str]:
    """
    Check for internal consistency in the configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of consistency errors, empty if all valid
    """
    errors = []
    try:
        # Load the config file
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            return ["Unsupported config file format. Use .yaml, .yml, or .json"]
            
        # Check language pair formats
        if 'training' in config_dict and 'language_pairs' in config_dict['training']:
            lang_pairs = config_dict['training']['language_pairs']
            for pair in lang_pairs:
                # Check if language pair is in the correct format (e.g., "en-es")
                if not isinstance(pair, str):
                    errors.append(f"Language pair must be a string, got {type(pair).__name__}: {pair}")
                    continue
                    
                if '-' not in pair and '_' not in pair:
                    errors.append(f"Invalid language pair format: {pair}. Expected format: 'source-target' or 'source_target'")
                    continue
                    
                separator = '-' if '-' in pair else '_'
                parts = pair.split(separator)
                
                if len(parts) != 2:
                    errors.append(f"Invalid language pair format: {pair}. Expected exactly two language codes separated by '{separator}'")
                    continue
                    
                source, target = parts
                if len(source) < 2 or len(target) < 2:
                    errors.append(f"Invalid language codes in pair: {pair}. Language codes should be at least 2 characters")
                    continue
        
        # Check training configuration
        if 'training' in config_dict:
            training = config_dict['training']
            
            # Check batch size vs. GPU memory
            if 'batch_size' in training and 'gpu_memory' in training:
                batch_size = training['batch_size']
                gpu_memory = training['gpu_memory']
                
                # Simple heuristic: 1GB GPU memory can handle batch size of ~4
                max_batch_size = gpu_memory * 4
                if batch_size > max_batch_size:
                    errors.append(f"Batch size ({batch_size}) may be too large for GPU memory ({gpu_memory}GB)")
            
            # Check learning rate range
            if 'learning_rate' in training:
                lr = training['learning_rate']
                if lr > 0.1:
                    errors.append(f"Learning rate ({lr}) is unusually high")
                elif lr < 1e-6:
                    errors.append(f"Learning rate ({lr}) is unusually low")
        
        # Check model configuration
        if 'model' in config_dict:
            model = config_dict['model']
            
            # Check encoder/decoder dimensions
            if 'encoder_dim' in model and 'decoder_dim' in model:
                encoder_dim = model['encoder_dim']
                decoder_dim = model['decoder_dim']
                
                if encoder_dim % 64 != 0:
                    errors.append(f"Encoder dimension ({encoder_dim}) is not a multiple of 64")
                if decoder_dim % 64 != 0:
                    errors.append(f"Decoder dimension ({decoder_dim}) is not a multiple of 64")
            
            # Check attention heads
            if 'num_heads' in model and 'encoder_dim' in model:
                num_heads = model['num_heads']
                encoder_dim = model['encoder_dim']
                
                if encoder_dim % num_heads != 0:
                    errors.append(f"Encoder dimension ({encoder_dim}) is not divisible by number of heads ({num_heads})")
        
        # Check deployment configuration
        if 'deployment' in config_dict:
            deployment = config_dict['deployment']
            
            # Check for conflicting deployment types
            if 'type' in deployment:
                deploy_type = deployment['type']
                if deploy_type not in ['docker', 'kubernetes', 'local']:
                    errors.append(f"Unknown deployment type: {deploy_type}")
                
                # Check for required configuration based on type
                if deploy_type == 'kubernetes' and 'kubernetes' not in deployment:
                    errors.append("Kubernetes deployment type specified but no kubernetes configuration found")
                elif deploy_type == 'docker' and 'docker' not in deployment:
                    errors.append("Docker deployment type specified but no docker configuration found")
        
        return errors
    except Exception as e:
        return [f"Error checking config consistency: {e}"]

def suggest_improvements(config_path: str) -> List[str]:
    """
    Suggest improvements to the configuration
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of suggestions
    """
    suggestions = []
    try:
        # Load the config file
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            return ["Unsupported config file format. Use .yaml, .yml, or .json"]
        
        # Check for missing sections
        if 'version' not in config_dict:
            suggestions.append("Add a 'version' field to track configuration versions")
        
        # Check for training optimizations
        if 'training' in config_dict:
            training = config_dict['training']
            
            # Check for mixed precision
            if 'mixed_precision' not in training:
                suggestions.append("Consider enabling mixed precision training for better performance")
            
            # Check for gradient accumulation
            if 'gradient_accumulation_steps' not in training:
                suggestions.append("Consider using gradient accumulation for larger effective batch sizes")
            
            # Check for learning rate scheduler
            if 'lr_scheduler' not in training:
                suggestions.append("Consider using a learning rate scheduler for better convergence")
        
        # Check for model optimizations
        if 'model' in config_dict:
            model = config_dict['model']
            
            # Check for quantization
            if 'quantization' not in model:
                suggestions.append("Consider enabling quantization for smaller model size")
            
            # Check for pruning
            if 'pruning' not in model:
                suggestions.append("Consider enabling pruning for smaller model size")
        
        # Check for monitoring
        if 'monitoring' not in config_dict:
            suggestions.append("Add monitoring configuration for better observability")
        
        # Check for logging
        if 'logging' not in config_dict:
            suggestions.append("Add logging configuration for better debugging")
        
        return suggestions
    except Exception as e:
        return [f"Error generating suggestions: {e}"]

def check_gpu_compatibility(config_path: str) -> Tuple[List[str], List[str]]:
    """
    Check if the configuration is compatible with the available GPUs
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        # Load the config file first to avoid undefined variable
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            errors.append("Unsupported config file format. Use .yaml, .yml, or .json")
            return errors, warnings
            
        # Try to import torch
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            if 'training' in config_dict and config_dict.get('training', {}).get('device') == 'cuda':
                errors.append("Configuration specifies CUDA but CUDA is not available")
            return errors, warnings
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        
        # Check training configuration
        if 'training' in config_dict:
            training = config_dict['training']
            
            # Check for distributed training
            if training.get('distributed', False) and gpu_count < 2:
                errors.append(f"Configuration specifies distributed training but only {gpu_count} GPU(s) available")
            
            # Check batch size vs. GPU memory
            if 'batch_size' in training:
                batch_size = training['batch_size']
                
                # Estimate GPU memory requirements based on model size and batch size
                model_size = config_dict.get('model', {}).get('encoder_dim', 1024) * config_dict.get('model', {}).get('decoder_dim', 1024) * 4 / (1024 * 1024 * 1024)  # in GB
                estimated_memory = model_size * batch_size * 4  # rough estimate
                
                # Check against available GPU memory
                for i, name in enumerate(gpu_names):
                    # Rough estimate of GPU memory based on name
                    if 'a100' in name.lower():
                        gpu_memory = 40  # A100 40GB
                    elif 'v100' in name.lower():
                        gpu_memory = 16  # V100 16GB
                    elif '3090' in name.lower():
                        gpu_memory = 24  # RTX 3090 24GB
                    elif '2080' in name.lower():
                        gpu_memory = 8   # RTX 2080 8GB
                    else:
                        gpu_memory = 8   # Default assumption
                    
                    if estimated_memory > gpu_memory * 0.8:
                        warnings.append(f"Batch size ({batch_size}) may be too large for GPU {i} ({name}) with estimated {gpu_memory}GB memory")
        
        return errors, warnings
    except ImportError:
        warnings.append("Could not import torch to check GPU compatibility")
        return errors, warnings
    except Exception as e:
        warnings.append(f"Error checking GPU compatibility: {e}")
        return errors, warnings

def main():
    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("--check-references", action="store_true", help="Check if referenced files exist")
    parser.add_argument("--check-consistency", action="store_true", help="Check for internal consistency")
    parser.add_argument("--suggest-improvements", action="store_true", help="Suggest improvements")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU compatibility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Validate schema
    print(f"🔍 Validating configuration file: {args.config_path}")
    schema_errors = validate_config(args.config_path)
    
    if schema_errors:
        print("❌ Schema validation errors:")
        for error in schema_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ Schema validation passed")
    
    # Check references if requested
    if args.check_references:
        reference_errors = check_config_references(args.config_path)
        if reference_errors:
            print("❌ Reference validation errors:")
            for error in reference_errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("✅ Reference validation passed")
    
    # Check consistency if requested
    if args.check_consistency:
        consistency_errors = check_config_consistency(args.config_path)
        if consistency_errors:
            print("⚠️ Consistency warnings:")
            for error in consistency_errors:
                print(f"  - {error}")
        else:
            print("✅ Consistency check passed")
    
    # Suggest improvements if requested
    if args.suggest_improvements:
        suggestions = suggest_improvements(args.config_path)
        if suggestions:
            print("\n💡 Suggestions for improvement:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
    
    # Check GPU compatibility if requested
    if args.check_gpu:
        gpu_errors, gpu_warnings = check_gpu_compatibility(args.config_path)
        
        if gpu_errors:
            print("❌ GPU compatibility errors:")
            for error in gpu_errors:
                print(f"  - {error}")
            sys.exit(1)
        
        if gpu_warnings:
            print("⚠️ GPU compatibility warnings:")
            for warning in gpu_warnings:
                print(f"  - {warning}")
        
        if not gpu_errors and not gpu_warnings:
            print("✅ GPU compatibility check passed")
    
    print("\n✅ Configuration is valid!")
    sys.exit(0)

if __name__ == "__main__":
    main()