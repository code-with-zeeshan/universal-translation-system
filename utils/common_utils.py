# utils/common_utils.py
"""
Common utilities for data pipeline - Low Priority Clean-up
Addresses: Directory creation, logging standardization, import cleanup
"""

import logging
from pathlib import Path
from typing import Optional, List
import sys
import os

class DirectoryManager:
    """Centralized directory creation and management"""
    
    @staticmethod
    def create_directory(path: str | Path, parents: bool = True, exist_ok: bool = True) -> Path:
        """
        Standardized directory creation
        
        Args:
            path: Directory path to create
            parents: Create parent directories if they don't exist
            exist_ok: Don't raise error if directory already exists
            
        Returns:
            Path object of created directory
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=parents, exist_ok=exist_ok)
        return path_obj
    
    @staticmethod
    def create_data_structure(base_dir: str = "data") -> dict:
        """
        Create standard data pipeline directory structure
        
        Returns:
            Dictionary with all created paths
        """
        base_path = Path(base_dir)
        
        directories = {
            'base': base_path,
            'log': base_path / 'log',
            'raw': base_path / 'raw',
            'essential': base_path / 'essential',
            'sampled': base_path / 'sampled',
            'final': base_path / 'final',
            'processed': base_path / 'processed',
            'opus': base_path / 'raw' / 'opus',
            'pivot_pairs': base_path / 'final' / 'pivot_pairs'
        }
        
        # Create all directories
        for name, path in directories.items():
            DirectoryManager.create_directory(path)
        
        return directories

class StandardLogger:
    """Standardized logging configuration for all pipeline components"""
    
    _loggers = {}  # Cache loggers to avoid reconfiguration
    
    @classmethod
    def get_logger(cls, name: str, log_dir: Optional[str] = None) -> logging.Logger:
        """
        Get standardized logger instance
        
        Args:
            name: Logger name (usually __name__)
            log_dir: Custom log directory (defaults to 'log')
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create log directory
        if log_dir is None:
            log_dir = 'log'
        DirectoryManager.create_directory(log_dir)
        
        # Configure logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(Path(log_dir) / 'data_pipeline.log')
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Standardized formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def log_system_info(cls, logger: logging.Logger) -> None:
        """Log standardized system information"""
        logger.info("=" * 60)
        logger.info("DATA PIPELINE SYSTEM INFO")
        logger.info("=" * 60)
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Available memory: {cls._get_memory_info()}")
        logger.info("=" * 60)
    
    @staticmethod
    def _get_memory_info() -> str:
        """Get memory information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total"
        except ImportError:
            return "Memory info unavailable (psutil not installed)"

class ImportCleaner:
    """Utility to identify and clean unused imports"""
    
    STANDARD_IMPORTS = {
        'pathlib': ['Path'],
        'logging': ['logging'],
        'typing': ['Dict', 'List', 'Optional', 'Tuple'],
        'dataclasses': ['dataclass'],
        'tqdm': ['tqdm'],
        'yaml': ['yaml'],
        'json': ['json'],
        'os': ['os'],
        'sys': ['sys']
    }
    
    ML_IMPORTS = {
        'datasets': ['load_dataset', 'Dataset'],
        'transformers': ['AutoModel', 'AutoTokenizer', 'pipeline'],
        'torch': ['torch'],
        'numpy': ['np'],
        'pandas': ['pd']
    }
    
    @classmethod
    def get_recommended_imports(cls, module_type: str = 'standard') -> List[str]:
        """
        Get recommended import statements
        
        Args:
            module_type: 'standard', 'ml', or 'all'
            
        Returns:
            List of import statements
        """
        imports = []
        
        if module_type in ['standard', 'all']:
            imports.extend(cls._format_imports(cls.STANDARD_IMPORTS))
        
        if module_type in ['ml', 'all']:
            imports.extend(cls._format_imports(cls.ML_IMPORTS))
        
        return imports
    
    @staticmethod
    def _format_imports(import_dict: dict) -> List[str]:
        """Format import dictionary to import statements"""
        formatted = []
        for module, items in import_dict.items():
            if len(items) == 1 and items[0] == module:
                formatted.append(f"import {module}")
            else:
                formatted.append(f"from {module} import {', '.join(items)}")
        return formatted

# Example usage and testing
if __name__ == "__main__":
    # Test directory creation
    print("Testing DirectoryManager...")
    dirs = DirectoryManager.create_data_structure("test_data")
    print(f"Created {len(dirs)} directories")
    
    # Test logging
    print("\nTesting StandardLogger...")
    logger = StandardLogger.get_logger(__name__)
    StandardLogger.log_system_info(logger)
    logger.info("✅ Test log message")
    logger.warning("⚠️  Test warning message")
    logger.error("❌ Test error message")
    
    # Test import recommendations
    print("\nTesting ImportCleaner...")
    standard_imports = ImportCleaner.get_recommended_imports('standard')
    print("Recommended standard imports:")
    for imp in standard_imports:
        print(f"  {imp}")
    
    print("\n✅ All utilities tested successfully!")