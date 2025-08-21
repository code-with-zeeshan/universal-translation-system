# utils/unified_validation.py
"""
Unified validation system for the Universal Translation System.
Combines input validation, configuration validation, and data validation.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import yaml

from pydantic import BaseModel, Field, validator, ValidationError
import torch

logger = logging.getLogger(__name__)


# ============= VALIDATION TYPES =============

class ValidationType(Enum):
    """Types of validation available"""
    INPUT = "input"
    CONFIG = "config"
    MODEL = "model"
    DATA = "data"
    PATH = "path"
    SYSTEM = "system"


class ValidationResult:
    """Result of validation with detailed information"""
    
    def __init__(self, valid: bool = True):
        self.valid = valid
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
    
    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)
    
    def __bool__(self) -> bool:
        """Allow using result as boolean"""
        return self.valid
    
    def __str__(self) -> str:
        """String representation"""
        if self.valid:
            return f"✅ Valid ({len(self.warnings)} warnings)"
        return f"❌ Invalid ({len(self.errors)} errors, {len(self.warnings)} warnings)"


# ============= INPUT VALIDATION (from validators.py) =============

class InputValidator:
    """Validate and sanitize user inputs"""
    
    # Regex patterns
    LANGUAGE_CODE_PATTERN = re.compile(r'^[a-z]{2,3}$', re.IGNORECASE)
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    
    # Limits
    MAX_TEXT_LENGTH = 5000
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    
    @staticmethod
    def validate_language_code(lang_code: str) -> ValidationResult:
        """Validate language code format"""
        result = ValidationResult()
        
        if not lang_code or not isinstance(lang_code, str):
            result.add_error("Language code must be a non-empty string")
            return result
        
        if not InputValidator.LANGUAGE_CODE_PATTERN.match(lang_code.lower()):
            result.add_error(f"Invalid language code format: {lang_code}")
            return result
        
        # Check against known language codes
        known_codes = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl', 'ru',
            'zh', 'ja', 'ko', 'ar', 'hi', 'tr', 'th', 'vi', 'id', 'uk'
        }
        
        if lang_code.lower() not in known_codes:
            result.add_warning(f"Unknown language code: {lang_code}")
        
        result.info['normalized'] = lang_code.lower()
        return result
    
    @staticmethod
    def validate_text_input(
        text: str, 
        max_length: Optional[int] = None,
        min_length: int = 1,
        allow_empty: bool = False
    ) -> Tuple[ValidationResult, str]:
        """
        Validate and sanitize text input.
        
        Returns:
            Tuple of (ValidationResult, sanitized_text)
        """
        result = ValidationResult()
        
        if not isinstance(text, str):
            result.add_error("Input must be a string")
            return result, ""
        
        # Remove control characters except newlines, tabs
        sanitized = ''.join(
            char for char in text 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # Check length
        if not allow_empty and len(sanitized.strip()) < min_length:
            result.add_error(f"Text too short (min: {min_length} characters)")
        
        max_len = max_length or InputValidator.MAX_TEXT_LENGTH
        if len(sanitized) > max_len:
            result.add_warning(f"Text truncated from {len(sanitized)} to {max_len} characters")
            sanitized = sanitized[:max_len]
        
        result.info['original_length'] = len(text)
        result.info['sanitized_length'] = len(sanitized)
        
        return result, sanitized.strip()
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[ValidationResult, str]:
        """
        Validate and sanitize filename.
        
        Returns:
            Tuple of (ValidationResult, sanitized_filename)
        """
        result = ValidationResult()
        
        if not filename or not isinstance(filename, str):
            result.add_error("Filename must be a non-empty string")
            return result, ""
        
        # Remove path components
        if '/' in filename or '\\' in filename:
            result.add_warning("Path components removed from filename")
            filename = Path(filename).name
        
        # Remove dangerous patterns
        sanitized = filename.replace('..', '').strip()
        
        # Check pattern
        if not InputValidator.FILENAME_PATTERN.match(sanitized):
            result.add_error(f"Invalid filename format: {sanitized}")
            return result, ""
        
        # Check length
        if len(sanitized) > InputValidator.MAX_FILENAME_LENGTH:
            result.add_error(f"Filename too long (max: {InputValidator.MAX_FILENAME_LENGTH})")
            return result, ""
        
        result.info['original'] = filename
        result.info['sanitized'] = sanitized
        
        return result, sanitized
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email address"""
        result = ValidationResult()
        
        if not email or not isinstance(email, str):
            result.add_error("Email must be a non-empty string")
            return result
        
        if not InputValidator.EMAIL_PATTERN.match(email.lower()):
            result.add_error(f"Invalid email format: {email}")
        
        return result
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate URL format"""
        result = ValidationResult()
        
        if not url or not isinstance(url, str):
            result.add_error("URL must be a non-empty string")
            return result
        
        if not InputValidator.URL_PATTERN.match(url):
            result.add_error(f"Invalid URL format: {url}")
            return result
        
        # Additional security checks
        if 'javascript:' in url.lower() or 'data:' in url.lower():
            result.add_error("Potentially dangerous URL scheme")
        
        return result


# ============= PATH VALIDATION =============

class PathValidator:
    """Validate and secure file paths"""
    
    @staticmethod
    def validate_path(
        path_str: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_dirs: Optional[List[str]] = None,
        create_if_missing: bool = False
    ) -> Tuple[ValidationResult, Optional[Path]]:
        """
        Comprehensive path validation.
        
        Returns:
            Tuple of (ValidationResult, resolved_path)
        """
        result = ValidationResult()
        
        try:
            path = Path(path_str).resolve()
        except Exception as e:
            result.add_error(f"Invalid path: {e}")
            return result, None
        
        # Security: Check if path is within allowed directories
        if allowed_dirs:
            allowed_paths = [Path(d).resolve() for d in allowed_dirs]
            if not any(
                str(path).startswith(str(allowed)) 
                for allowed in allowed_paths
            ):
                result.add_error(f"Path outside allowed directories: {path}")
                return result, None
        
        # Existence checks
        if must_exist and not path.exists():
            if create_if_missing and must_be_dir:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result.add_warning(f"Created directory: {path}")
                except Exception as e:
                    result.add_error(f"Cannot create directory: {e}")
                    return result, None
            else:
                result.add_error(f"Path does not exist: {path}")
                return result, None
        
        # Type checks
        if must_be_file and path.exists() and not path.is_file():
            result.add_error(f"Path is not a file: {path}")
            return result, None
        
        if must_be_dir and path.exists() and not path.is_dir():
            result.add_error(f"Path is not a directory: {path}")
            return result, None
        
        # Permission checks
        if path.exists():
            if not os.access(path, os.R_OK):
                result.add_warning(f"No read permission: {path}")
            
            if must_be_file or must_be_dir:
                parent = path.parent if must_be_file else path
                if not os.access(parent, os.W_OK):
                    result.add_warning(f"No write permission: {parent}")
        
        result.info['resolved_path'] = str(path)
        result.info['exists'] = path.exists()
        
        return result, path


# ============= CONFIGURATION VALIDATION =============

class ConfigValidator:
    """Enhanced configuration validation using Pydantic schemas"""
    
    @staticmethod
    def validate_config(
        config_data: Union[Dict, str, Path],
        schema_class: Optional[BaseModel] = None
    ) -> Tuple[ValidationResult, Optional[BaseModel]]:
        """
        Validate configuration against schema.
        
        Args:
            config_data: Configuration dict, YAML string, or path to YAML file
            schema_class: Pydantic model class to validate against
            
        Returns:
            Tuple of (ValidationResult, validated_config)
        """
        result = ValidationResult()
        
        # Load configuration if needed
        if isinstance(config_data, (str, Path)):
            try:
                if Path(config_data).exists():
                    # It's a file path
                    with open(config_data, 'r') as f:
                        config_dict = yaml.safe_load(f)
                else:
                    # It's a YAML string
                    config_dict = yaml.safe_load(config_data)
            except Exception as e:
                result.add_error(f"Failed to load configuration: {e}")
                return result, None
        else:
            config_dict = config_data
        
        # Use default schema if not provided
        if schema_class is None:
            from config.schemas import RootConfig
            schema_class = RootConfig
        
        # Validate against schema
        try:
            validated_config = schema_class(**config_dict)
            result.info['config_class'] = schema_class.__name__
            return result, validated_config
        except ValidationError as e:
            for error in e.errors():
                field = '.'.join(str(f) for f in error['loc'])
                msg = error['msg']
                result.add_error(f"{field}: {msg}")
            return result, None
        except Exception as e:
            result.add_error(f"Validation failed: {e}")
            return result, None
    
    @staticmethod
    def validate_required_paths(config: Any) -> ValidationResult:
        """Validate that required paths in config exist"""
        result = ValidationResult()
        
        # Define required paths based on config structure
        path_fields = [
            ('data.processed_dir', True, True),  # (field, must_exist, is_dir)
            ('data.checkpoint_dir', False, True),
            ('vocabulary.vocab_dir', True, True),
        ]
        
        for field_path, must_exist, is_dir in path_fields:
            # Navigate to field value
            obj = config
            for field in field_path.split('.'):
                if hasattr(obj, field):
                    obj = getattr(obj, field)
                else:
                    result.add_warning(f"Config field not found: {field_path}")
                    obj = None
                    break
            
            if obj:
                path_result, _ = PathValidator.validate_path(
                    obj,
                    must_exist=must_exist,
                    must_be_dir=is_dir,
                    create_if_missing=True
                )
                
                if not path_result.valid:
                    for error in path_result.errors:
                        result.add_error(f"{field_path}: {error}")
                
                for warning in path_result.warnings:
                    result.add_warning(f"{field_path}: {warning}")
        
        return result
    
    @staticmethod
    def create_default_config(
        output_path: str = "config/default.yaml",
        tier: str = "basic"
    ) -> Path:
        """Create a default configuration file"""
        
        configs = {
            'basic': {
                'data': {
                    'processed_dir': 'data/processed',
                    'checkpoint_dir': 'checkpoints',
                    'training_distribution': {
                        'en-es': 100000,
                        'en-fr': 100000,
                        'en-de': 100000
                    },
                    'quality_threshold': 0.8,
                    'total_size_gb': 1.0
                },
                'model': {
                    'vocab_size': 32000,
                    'hidden_dim': 512,
                    'num_layers': 4,
                    'num_heads': 8
                },
                'training': {
                    'batch_size': 16,
                    'learning_rate': 0.0001,
                    'num_epochs': 10
                },
                'memory': {
                    'mixed_precision': False,
                    'gradient_checkpointing': False
                },
                'vocabulary': {
                    'vocab_dir': 'vocabs',
                    'dynamic_vocabulary': False
                }
            },
            'advanced': {
                'data': {
                    'processed_dir': 'data/processed',
                    'checkpoint_dir': 'checkpoints',
                    'training_distribution': {
                        'en-es': 1000000,
                        'en-fr': 1000000,
                        'en-de': 1000000,
                        'en-zh': 500000,
                        'en-ja': 500000
                    },
                    'quality_threshold': 0.9,
                    'total_size_gb': 10.0
                },
                'model': {
                    'vocab_size': 50000,
                    'hidden_dim': 1024,
                    'num_layers': 6,
                    'num_heads': 16
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.0005,
                    'num_epochs': 20,
                    'use_fsdp': True
                },
                'memory': {
                    'mixed_precision': True,
                    'gradient_checkpointing': True,
                    'flash_attention': True
                },
                'vocabulary': {
                    'vocab_dir': 'vocabs',
                    'dynamic_vocabulary': True,
                    'vocab_switch_penalty': 0.001
                }
            }
        }
        
        config = configs.get(tier, configs['basic'])
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created {tier} configuration at {output}")
        return output


# ============= MODEL VALIDATION =============

class ModelValidator:
    """Validate model files and checkpoints"""
    
    @staticmethod
    def validate_checkpoint(
        checkpoint_path: str,
        expected_keys: Optional[List[str]] = None,
        check_device: bool = True
    ) -> ValidationResult:
        """Validate a model checkpoint file"""
        result = ValidationResult()
        
        # Validate path
        path_result, path = PathValidator.validate_path(
            checkpoint_path,
            must_exist=True,
            must_be_file=True
        )
        
        if not path_result.valid:
            result.merge(path_result)
            return result
        
        # Try to load checkpoint
        try:
            if check_device and torch.cuda.is_available():
                checkpoint = torch.load(path, map_location='cuda')
            else:
                checkpoint = torch.load(path, map_location='cpu')
        except Exception as e:
            result.add_error(f"Cannot load checkpoint: {e}")
            return result
        
        # Check expected keys
        if expected_keys:
            if isinstance(checkpoint, dict):
                missing_keys = [
                    key for key in expected_keys 
                    if key not in checkpoint
                ]
                if missing_keys:
                    result.add_error(f"Missing keys in checkpoint: {missing_keys}")
            else:
                result.add_warning("Checkpoint is not a dictionary")
        
        # Get checkpoint info
        if isinstance(checkpoint, dict):
            result.info['keys'] = list(checkpoint.keys())
            
            # Check for common keys
            if 'model_state_dict' in checkpoint:
                result.info['model_params'] = len(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                result.info['model_params'] = len(checkpoint['state_dict'])
            
            if 'epoch' in checkpoint:
                result.info['epoch'] = checkpoint['epoch']
            
            if 'loss' in checkpoint:
                result.info['loss'] = checkpoint['loss']
        
        result.info['file_size_mb'] = path.stat().st_size / (1024 * 1024)
        
        return result
    
    @staticmethod
    def validate_model_source(source: str) -> ValidationResult:
        """Validate model source for security"""
        result = ValidationResult()
        
        # Define trusted sources
        trusted_sources = [
            'facebook/',
            'microsoft/',
            'google/',
            'Helsinki-NLP/',
            'openai/',
            'EleutherAI/',
            'bigscience/'
        ]
        
        if not any(source.startswith(trusted) for trusted in trusted_sources):
            result.add_warning(f"Model from untrusted source: {source}")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            'eval(',
            'exec(',
            '__import__',
            'os.system',
            'subprocess'
        ]
        
        for pattern in suspicious_patterns:
            if pattern in source:
                result.add_error(f"Suspicious pattern in model source: {pattern}")
                break
        
        return result


# ============= DATA VALIDATION =============

class DataValidator:
    """Validate data files and datasets"""
    
    @staticmethod
    def validate_dataset_file(
        file_path: str,
        expected_format: str = 'tsv',
        sample_size: int = 100,
        check_encoding: bool = True
    ) -> ValidationResult:
        """Validate a dataset file"""
        result = ValidationResult()
        
        # Validate path
        path_result, path = PathValidator.validate_path(
            file_path,
            must_exist=True,
            must_be_file=True
        )
        
        if not path_result.valid:
            result.merge(path_result)
            return result
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        result.info['file_size_mb'] = file_size_mb
        
        if file_size_mb == 0:
            result.add_error("File is empty")
            return result
        
        if file_size_mb > 10000:  # 10GB
            result.add_warning(f"Very large file: {file_size_mb:.1f}MB")
        
        # Check encoding and format
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1'] if check_encoding else ['utf-8']
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= sample_size:
                                break
                            lines.append(line.strip())
                        
                        result.info['encoding'] = encoding
                        break
                except UnicodeDecodeError:
                    continue
            else:
                result.add_error("Cannot decode file with standard encodings")
                return result
            
            # Validate format
            if expected_format == 'tsv':
                invalid_lines = []
                for i, line in enumerate(lines):
                    if line and '\t' not in line:
                        invalid_lines.append(i + 1)
                
                if invalid_lines:
                    result.add_error(
                        f"Invalid TSV format on lines: {invalid_lines[:10]}"
                    )
            
            elif expected_format == 'jsonl':
                import json
                invalid_lines = []
                for i, line in enumerate(lines):
                    if line:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            invalid_lines.append(i + 1)
                
                if invalid_lines:
                    result.add_error(
                        f"Invalid JSONL format on lines: {invalid_lines[:10]}"
                    )
            
            result.info['sample_lines'] = len(lines)
            result.info['format'] = expected_format
            
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
        
        return result
    
    @staticmethod
    def validate_language_pair(
        file_path: str,
        source_lang: str,
        target_lang: str,
        sample_size: int = 100
    ) -> ValidationResult:
        """Validate a parallel corpus file"""
        result = ValidationResult()
        
        # First validate as dataset file
        file_result = DataValidator.validate_dataset_file(
            file_path,
            expected_format='tsv',
            sample_size=sample_size
        )
        
        if not file_result.valid:
            return file_result
        
        # Additional validation for language pairs
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_lengths = []
                empty_sources = 0
                empty_targets = 0
                
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        src, tgt = parts
                        
                        if not src.strip():
                            empty_sources += 1
                        if not tgt.strip():
                            empty_targets += 1
                        
                        line_lengths.append((len(src.split()), len(tgt.split())))
                
                # Check for empty fields
                if empty_sources > 0:
                    result.add_warning(f"Found {empty_sources} empty source sentences")
                if empty_targets > 0:
                    result.add_warning(f"Found {empty_targets} empty target sentences")
                
                # Check length ratios
                suspicious_ratios = 0
                for src_len, tgt_len in line_lengths:
                    if src_len > 0 and tgt_len > 0:
                        ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
                        if ratio > 3:  # One sentence is 3x longer
                            suspicious_ratios += 1
                
                if suspicious_ratios > sample_size * 0.1:  # >10% suspicious
                    result.add_warning(
                        f"Many suspicious length ratios: {suspicious_ratios}/{len(line_lengths)}"
                    )
                
                result.info['source_lang'] = source_lang
                result.info['target_lang'] = target_lang
                result.info['sampled_pairs'] = len(line_lengths)
                
        except Exception as e:
            result.add_error(f"Error validating language pair: {e}")
        
        return result


# ============= SYSTEM VALIDATION =============

class SystemValidator:
    """Validate entire system setup"""
    
    @staticmethod
    def validate_environment() -> ValidationResult:
        """Validate system environment and dependencies"""
        result = ValidationResult()
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        result.info['python_version'] = python_version
        
        if sys.version_info < (3, 8):
            result.add_error(f"Python 3.8+ required, found {python_version}")
        
        # Check required packages
        required_packages = {
            'torch': '2.0.0',
            'transformers': '4.0.0',
            'pydantic': '2.0.0',
            'numpy': '1.19.0',
            'yaml': None,  # Any version
            'tqdm': None
        }
        
        import importlib
        import pkg_resources
        
        for package, min_version in required_packages.items():
            try:
                mod = importlib.import_module(package.replace('-', '_'))
                
                if min_version and hasattr(mod, '__version__'):
                    installed_version = mod.__version__
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                        result.add_warning(
                            f"{package} version {installed_version} < {min_version}"
                        )
                
                result.info[f'{package}_version'] = getattr(mod, '__version__', 'unknown')
                
            except ImportError:
                result.add_error(f"Required package not found: {package}")
        
        # Check CUDA availability
        try:
            import torch
            result.info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result.info['cuda_version'] = torch.version.cuda
                result.info['gpu_count'] = torch.cuda.device_count()
                result.info['gpu_names'] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        except Exception as e:
            result.add_warning(f"Cannot check CUDA: {e}")
        
        # Check disk space
        import shutil
        for path in ['/', '/tmp', '.']:
            try:
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024**3)
                result.info[f'disk_free_{path}'] = f"{free_gb:.1f}GB"
                
                if free_gb < 10:
                    result.add_warning(f"Low disk space on {path}: {free_gb:.1f}GB")
                    
            except Exception:
                pass
        
        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            result.info['ram_total_gb'] = f"{mem.total / (1024**3):.1f}"
            result.info['ram_available_gb'] = f"{mem.available / (1024**3):.1f}"
            
            if mem.available < 4 * (1024**3):  # Less than 4GB
                result.add_warning(f"Low memory: {mem.available / (1024**3):.1f}GB available")
                
        except ImportError:
            pass
        
        return result
    
    @staticmethod
    def validate_full_system(config_path: str) -> ValidationResult:
        """Complete system validation"""
        result = ValidationResult()
        
        # 1. Validate environment
        env_result = SystemValidator.validate_environment()
        result.merge(env_result)
        
        # 2. Validate configuration
        from config.schemas import RootConfig
        config_result, config = ConfigValidator.validate_config(
            config_path,
            RootConfig
        )
        result.merge(config_result)
        
        if config:
            # 3. Validate paths in config
            path_result = ConfigValidator.validate_required_paths(config)
            result.merge(path_result)
        
        # 4. Check for model files
        model_paths = [
            'models/production/encoder.pt',
            'models/production/decoder.pt'
        ]
        
        for model_path in model_paths:
            if Path(model_path).exists():
                model_result = ModelValidator.validate_checkpoint(model_path)
                if not model_result.valid:
                    result.add_warning(f"Invalid model at {model_path}")
            else:
                result.info[f'{Path(model_path).stem}_exists'] = False
        
        return result


# ============= UNIFIED VALIDATOR =============

class UnifiedValidator:
    """
    Main entry point for all validation needs.
    Provides a unified interface to all validators.
    """
    
    def __init__(self):
        """Initialize unified validator"""
        self.input = InputValidator()
        self.path = PathValidator()
        self.config = ConfigValidator()
        self.model = ModelValidator()
        self.data = DataValidator()
        self.system = SystemValidator()
    
    def validate(
        self,
        validation_type: ValidationType,
        target: Any,
        **kwargs
    ) -> ValidationResult:
        """
        Main validation method routing to appropriate validator.
        
        Args:
            validation_type: Type of validation to perform
            target: Object to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult object
        """
        if validation_type == ValidationType.INPUT:
            # Route to input validation
            if 'language_code' in kwargs:
                return self.input.validate_language_code(target)
            elif 'email' in kwargs:
                return self.input.validate_email(target)
            elif 'url' in kwargs:
                return self.input.validate_url(target)
            else:
                result, _ = self.input.validate_text_input(target, **kwargs)
                return result
                
        elif validation_type == ValidationType.PATH:
            result, _ = self.path.validate_path(target, **kwargs)
            return result
            
        elif validation_type == ValidationType.CONFIG:
            result, _ = self.config.validate_config(target, **kwargs)
            return result
            
        elif validation_type == ValidationType.MODEL:
            return self.model.validate_checkpoint(target, **kwargs)
            
        elif validation_type == ValidationType.DATA:
            return self.data.validate_dataset_file(target, **kwargs)
            
        elif validation_type == ValidationType.SYSTEM:
            return self.system.validate_full_system(target)
            
        else:
            result = ValidationResult(valid=False)
            result.add_error(f"Unknown validation type: {validation_type}")
            return result
    
    def validate_batch(
        self,
        items: List[Tuple[ValidationType, Any, Dict]]
    ) -> Dict[int, ValidationResult]:
        """
        Validate multiple items.
        
        Args:
            items: List of (validation_type, target, kwargs) tuples
            
        Returns:
            Dictionary mapping index to ValidationResult
        """
        results = {}
        
        for i, (val_type, target, kwargs) in enumerate(items):
            results[i] = self.validate(val_type, target, **kwargs)
        
        return results
    
    def get_summary(
        self,
        results: Union[ValidationResult, List[ValidationResult], Dict[Any, ValidationResult]]
    ) -> str:
        """
        Generate a summary of validation results.
        
        Args:
            results: Single result, list, or dict of results
            
        Returns:
            Formatted summary string
        """
        summary_lines = ["Validation Summary", "=" * 50]
        
        if isinstance(results, ValidationResult):
            results = [results]
        elif isinstance(results, dict):
            results = list(results.values())
        
        total = len(results)
        valid = sum(1 for r in results if r.valid)
        errors = sum(len(r.errors) for r in results)
        warnings = sum(len(r.warnings) for r in results)
        
        summary_lines.append(f"Total validations: {total}")
        summary_lines.append(f"Valid: {valid}/{total} ({valid/total*100:.1f}%)")
        summary_lines.append(f"Total errors: {errors}")
        summary_lines.append(f"Total warnings: {warnings}")
        
        if errors > 0:
            summary_lines.append("\nErrors:")
            for i, result in enumerate(results):
                for error in result.errors:
                    summary_lines.append(f"  [{i}] {error}")
        
        if warnings > 0:
            summary_lines.append("\nWarnings:")
            for i, result in enumerate(results):
                for warning in result.warnings:
                    summary_lines.append(f"  [{i}] {warning}")
        
        return "\n".join(summary_lines)


# ============= CONVENIENCE FUNCTIONS =============

def quick_validate(target: Any, val_type: str = 'auto', **kwargs) -> bool:
    """
    Quick validation helper that returns boolean.
    
    Args:
        target: Object to validate
        val_type: Validation type or 'auto' to detect
        **kwargs: Additional parameters
        
    Returns:
        True if valid, False otherwise
    """
    validator = UnifiedValidator()
    
    if val_type == 'auto':
        # Auto-detect type
        if isinstance(target, (str, Path)) and Path(target).suffix in ['.yaml', '.yml']:
            val_type = 'config'
        elif isinstance(target, (str, Path)) and Path(target).suffix in ['.pt', '.pth']:
            val_type = 'model'
        elif isinstance(target, str) and len(target) <= 3:
            val_type = 'language_code'
        elif isinstance(target, str) and '@' in target:
            val_type = 'email'
        elif isinstance(target, str) and target.startswith('http'):
            val_type = 'url'
        else:
            val_type = 'text'
    
    type_map = {
        'config': ValidationType.CONFIG,
        'model': ValidationType.MODEL,
        'data': ValidationType.DATA,
        'path': ValidationType.PATH,
        'system': ValidationType.SYSTEM,
        'text': ValidationType.INPUT,
        'language_code': ValidationType.INPUT,
        'email': ValidationType.INPUT,
        'url': ValidationType.INPUT
    }
    
    validation_type = type_map.get(val_type, ValidationType.INPUT)
    
    if val_type in ['language_code', 'email', 'url']:
        kwargs[val_type] = True
    
    result = validator.validate(validation_type, target, **kwargs)
    return result.valid


def validate_and_report(target: Any, val_type: str = 'auto', **kwargs) -> ValidationResult:
    """
    Validate and print report.
    
    Args:
        target: Object to validate
        val_type: Validation type
        **kwargs: Additional parameters
        
    Returns:
        ValidationResult object
    """
    validator = UnifiedValidator()
    
    # Map string to enum
    type_map = {
        'input': ValidationType.INPUT,
        'config': ValidationType.CONFIG,
        'model': ValidationType.MODEL,
        'data': ValidationType.DATA,
        'path': ValidationType.PATH,
        'system': ValidationType.SYSTEM
    }
    
    validation_type = type_map.get(val_type, ValidationType.INPUT)
    result = validator.validate(validation_type, target, **kwargs)
    
    # Print report
    print(f"\n{result}")
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    if result.info:
        print("\nInfo:")
        for key, value in result.info.items():
            print(f"  ℹ️  {key}: {value}")
    
    return result


def main():
    """Example usage and testing"""
    import sys
    
    print("Universal Translation System - Unified Validator")
    print("=" * 50)
    
    validator = UnifiedValidator()
    
    # Example: Validate system
    print("\n1. System Validation:")
    result = validator.system.validate_environment()
    print(f"Result: {result}")
    
    # Example: Validate configuration
    if len(sys.argv) > 1:
        print(f"\n2. Config Validation ({sys.argv[1]}):")
        result = validate_and_report(sys.argv[1], 'config')
    
    # Example: Validate language code
    print("\n3. Language Code Validation:")
    for code in ['en', 'eng', 'xx', '123', '']:
        result = quick_validate(code, 'language_code')
        print(f"  {code}: {'✅' if result else '❌'}")
    
    # Example: Validate text
    print("\n4. Text Validation:")
    text = "Hello\x00World\x01Test"  # Text with control characters
    result, clean = validator.input.validate_text_input(text)
    print(f"  Original: {repr(text)}")
    print(f"  Cleaned: {repr(clean)}")
    print(f"  Valid: {result.valid}")


if __name__ == "__main__":
    main()