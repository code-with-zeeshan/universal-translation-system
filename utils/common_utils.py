# utils/common_utils.py
"""
Common utilities for data pipeline - Low Priority Clean-up
Addresses: Directory creation, import cleanup
"""

import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from config.schemas import RootConfig
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
            'raw': base_path / 'raw',
            'processed': processed_path,
            'corpus': processed_path / 'corpus',
            'sampled': processed_path / 'sampled',
            'final': processed_path / 'augment',
            'opus': base_path / 'raw' / 'opus',
            'pivot_pairs': processed_path / 'augment' / 'pivot_pairs'
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


class RuntimeDirectoryManager:
    """Centralized source of truth for ALL runtime file/directory paths.
    
    Every module that reads or writes runtime artifacts should obtain
    paths through this class to ensure consistency across the system.
    
    Usage:
        mgr = RuntimeDirectoryManager(root=".")
        train_path = mgr.train_final_path
        mgr.ensure_data_structure()
    """

    def __init__(self, config: Optional['RootConfig'] = None, root: Union[str, Path] = "output"):
        self.root = Path(root).resolve()
        self.config = config

    def _resolve(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def _cfg(self, attr: str, default: str) -> Path:
        if self.config is not None:
            val = getattr(self.config, attr, None)
            if val is not None:
                return self.root / str(val)
        return self._resolve(*default.split("/"))

    # ── Data Pipeline ────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return self._resolve("data")

    @property
    def raw_dir(self) -> Path:
        return self._resolve("data", "raw")

    @property
    def opus_dir(self) -> Path:
        return self._resolve("data", "raw", "opus")

    @property
    def processed_dir(self) -> Path:
        return self._cfg("processed_dir", "data/processed")

    @property
    def sampled_dir(self) -> Path:
        return self.processed_dir / "sampled"

    @property
    def augment_dir(self) -> Path:
        return self.processed_dir / "augment"

    @property
    def pivot_pairs_dir(self) -> Path:
        return self.augment_dir / "pivot_pairs"

    @property
    def distilled_dir(self) -> Path:
        return self.augment_dir / "distilled"

    @property
    def corpus_dir(self) -> Path:
        return self.processed_dir / "corpus"

    @property
    def cache_dir(self) -> Path:
        return self._cfg("cache_dir", "data/processed/cache")

    @property
    def eval_data_dir(self) -> Path:
        return self._resolve("data", "evaluation")

    @property
    def datasets_dir(self) -> Path:
        return self._resolve("datasets")

    @property
    def train_final_path(self) -> Path:
        return self.datasets_dir / "train_final.txt"

    @property
    def val_final_path(self) -> Path:
        return self.datasets_dir / "val_final.txt"

    @property
    def train_temp_path(self) -> Path:
        return self.processed_dir / "train_temp.txt"

    # ── Training ─────────────────────────────────────────────────
    @property
    def checkpoints_dir(self) -> Path:
        return self._cfg("checkpoint_dir", "checkpoints")

    # ── Models ───────────────────────────────────────────────────
    @property
    def models_dir(self) -> Path:
        return self._resolve("models")

    @property
    def encoder_models_dir(self) -> Path:
        return self._resolve("models", "encoder")

    @property
    def decoder_models_dir(self) -> Path:
        return self._resolve("models", "decoder")

    @property
    def adapters_dir(self) -> Path:
        return self._resolve("models", "adapters")

    @property
    def production_dir(self) -> Path:
        return self._resolve("models", "production")

    @property
    def model_registry_path(self) -> Path:
        return self.models_dir / "model_registry.json"

    # ── Vocabulary ───────────────────────────────────────────────
    @property
    def vocab_dir(self) -> Path:
        return self._cfg("vocab_dir", "vocabulary/vocab")

    @property
    def vocab_manifest_path(self) -> Path:
        return self.vocab_dir / "manifest.json"

    # ── Logs ─────────────────────────────────────────────────────
    @property
    def logs_dir(self) -> Path:
        return self._resolve("logs")

    # ── Evaluation ───────────────────────────────────────────────
    @property
    def eval_reports_dir(self) -> Path:
        return self._resolve("evaluation_reports")

    # ── Training Visualizations ──────────────────────────────────
    @property
    def training_viz_dir(self) -> Path:
        return self._resolve("training_visualizations")

    # ── Profiling ────────────────────────────────────────────────
    @property
    def profiles_dir(self) -> Path:
        return self._resolve("profiles")

    # ── Generated Configs ────────────────────────────────────────
    @property
    def generated_config_dir(self) -> Path:
        return self._resolve("config")

    # ── Pipeline / Runtime State ─────────────────────────────────
    @property
    def pipeline_state_path(self) -> Path:
        return self._resolve("pipeline_state.json")

    @property
    def streaming_eval_cache_path(self) -> Path:
        return self._resolve("streaming_evaluation_cache.json")

    @property
    def decoder_pool_path(self) -> Path:
        return self.generated_config_dir / "decoder_pool.json"

    @property
    def version_config_path(self) -> Path:
        return self._resolve("version-config.json")

    # ── Domain data ──────────────────────────────────────────────
    @property
    def medical_dir(self) -> Path:
        return self.raw_dir / "medical"

    @property
    def legal_dir(self) -> Path:
        return self.raw_dir / "legal"

    @property
    def tech_dir(self) -> Path:
        return self.raw_dir / "tech"

    # ── Directory Creation ───────────────────────────────────────
    def ensure(self, *paths: Path) -> None:
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)

    def ensure_data_structure(self) -> dict:
        self.ensure(
            self.raw_dir, self.opus_dir, self.processed_dir,
            self.sampled_dir, self.augment_dir, self.pivot_pairs_dir,
            self.distilled_dir, self.corpus_dir, self.cache_dir,
            self.eval_data_dir, self.datasets_dir,
        )
        return {
            'base': self.data_dir,
            'raw': self.raw_dir,
            'processed': self.processed_dir,
            'corpus': self.corpus_dir,
            'sampled': self.sampled_dir,
            'final': self.augment_dir,
            'opus': self.opus_dir,
            'pivot_pairs': self.pivot_pairs_dir,
            'datasets': self.datasets_dir,
        }

    def ensure_all(self) -> None:
        self.ensure_data_structure()
        self.ensure(
            self.checkpoints_dir,
            self.encoder_models_dir, self.decoder_models_dir,
            self.adapters_dir, self.production_dir,
            self.vocab_dir, self.logs_dir, self.eval_reports_dir,
            self.training_viz_dir, self.profiles_dir,
        )

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
