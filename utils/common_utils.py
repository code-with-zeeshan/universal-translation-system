# utils/common_utils.py
"""
Common utilities for data pipeline - Low Priority Clean-up
Addresses: Directory creation, import cleanup
"""

import logging
from pathlib import Path
from typing import Optional, List, Union, Dict
import sys
import os

class DirectoryManager:
    """Centralized directory creation and management"""
    
    @staticmethod
    def create_directory(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
        """
        Standardized directory creation with better error handling
        
        Args:
            path: Directory path to create
            parents: Create parent directories if they don't exist
            exist_ok: Don't raise error if directory already exists
            
        Returns:
            Path object of created directory
        """
        path_obj = Path(path)
        try:
            path_obj.mkdir(parents=parents, exist_ok=exist_ok)
            return path_obj
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot create directory {path_obj}")
        except OSError as e:
            raise OSError(f"Failed to create directory {path_obj}: {e}")    
    
    @staticmethod
    def create_data_structure(base_dir: str = "data") -> dict:
        """
        Create standard data pipeline directory structure
        
        Returns:
            Dictionary with all created paths
        """
        base_path = Path(base_dir)
        processed_path = base_path / 'processed'
        
        directories = {
            'base': base_path,
            'log': base_path / 'log',
            'raw': base_path / 'raw',
            'essential': base_path / 'essential',
            'processed': processed_path,
            'sampled': processed_path / 'sampled',
            'final': processed_path / 'final',
            'ready': processed_path / 'ready',
            'opus': base_path / 'raw' / 'opus',
            'pivot_pairs': processed_path / 'final' / 'pivot_pairs'
        }
        
        # Create all directories
        for name, path in directories.items():
            DirectoryManager.create_directory(path)
        
        return directories
    
    @staticmethod
    def create_logs_structure(base_dir: str = "logs") -> Dict[str, Path]:
        """
        Create standard logs directory structure at repo root.
        Sections include: data, training, monitoring, coordinator, decoder, evaluation, vocabulary.
        Returns a dict of created paths.
        """
        base_path = Path(base_dir)
        sections = [
            'data',
            'training',
            'monitoring',
            'coordinator',
            'decoder',
            'evaluation',
            'vocabulary',
        ]
        paths: Dict[str, Path] = {'base': base_path}
        for section in sections:
            p = base_path / section
            p.mkdir(parents=True, exist_ok=True)
            paths[section] = p
        return paths

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
