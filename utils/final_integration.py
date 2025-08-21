# utils/final_integration.py
"""
Final integration utilities to ensure all components work together
"""
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SystemIntegrator:
    """Ensure all system components are properly integrated"""
    
    @staticmethod
    def validate_imports() -> Dict[str, bool]:
        """Validate all required imports are available"""
        required_modules = {
            # Utils
            'utils.security': ['validate_model_source', 'safe_load_model'],
            'utils.base_classes': ['BaseDataProcessor', 'TokenizerMixin'],
            'utils.dataset_classes': ['ModernParallelDataset'],
            'utils.common_utils': ['DirectoryManager'], # Removed StandardLogger
            
            # Data
            'data.data_utils': ['ConfigManager', 'DataProcessor'],
            'data.pipeline_connector': ['PipelineConnector'],
            'data.vocabulary_connector': ['VocabularyConnector'],
            
            # Vocabulary
            'vocabulary.unified_vocab_manager': ['UnifiedVocabularyManager', 'VocabularyMode'],
            
            # Training
            'training.distributed_train': ['UnifiedDistributedTrainer'],
            'training.memory_efficient_training': ['MemoryOptimizedTrainer'],
            'training.progressive_training': ['ProgressiveTrainingOrchestrator'], # Corrected class name
            'training.quantization_pipeline': ['EncoderQuantizer'],
            
            # Evaluation
            'evaluation.evaluate_model': ['TranslationEvaluator'],
            
            # Integration
            'integration.connect_all_systems': ['UniversalTranslationSystem'],
        }
        
        results = {}
        
        for module_name, required_attrs in required_modules.items():
            try:
                module = importlib.import_module(module_name)
                
                # Check required attributes
                missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
                
                if missing_attrs:
                    results[module_name] = False
                    logger.error(f"Module {module_name} missing attributes: {missing_attrs}")
                else:
                    results[module_name] = True
                    
            except ImportError as e:
                results[module_name] = False
                logger.error(f"Failed to import {module_name}: {e}")
        
        return results
    
    @staticmethod
    def fix_python_path():
        """Ensure all directories are in Python path"""
        project_root = Path(__file__).parent.parent
        
        # Add all necessary directories to path
        directories = [
            project_root,
            project_root / 'data',
            project_root / 'utils',
            project_root / 'vocabulary',
            project_root / 'training',
            project_root / 'evaluation',
            project_root / 'integration',
            project_root / 'encoder',
            project_root / 'cloud_decoder',
        ]
        
        for directory in directories:
            if directory.exists() and str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
                logger.info(f"Added to Python path: {directory}")
    
    @staticmethod
    def create_missing_directories():
        """Create all required directories"""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            'data/raw',
            'data/processed',
            'data/essential',
            'data/sampled',
            'data/final',
            'data/cache',
            'models/production',
            'models/encoder',
            'models/decoder',
            'vocabs',
            'checkpoints',
            'logs',
            'config',
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {full_path}")
    
    @staticmethod
    def validate_system_ready() -> bool:
        """Check if system is ready to run"""
        # Fix Python path first
        SystemIntegrator.fix_python_path()
        
        # Create directories
        SystemIntegrator.create_missing_directories()
        
        # Validate imports
        import_results = SystemIntegrator.validate_imports()
        
        # Check critical files
        project_root = Path(__file__).parent.parent
        critical_files = [
            'data/config.yaml',
            'config/base.yaml',
        ]
        
        files_exist = all((project_root / f).exists() for f in critical_files)
        
        if all_imports_ok and files_exist:
            logger.info("✅ System validation passed - ready to run!")
            return True
        else:
            logger.error("❌ System validation failed")
            if not all_imports_ok:
                failed_imports = [k for k, v in import_results.items() if not v]
                logger.error(f"Failed imports: {failed_imports}")
            if not files_exist:
                logger.error("Critical configuration files missing")
            return False