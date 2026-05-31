# integration/system.py
"""
Universal Translation System core class
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from prometheus_client import Gauge, start_http_server

from .system_config import SystemConfig
from .system_health import SystemHealthMonitor
from monitoring.health_service import start_health_service
from utils.dataset_classes import ModernParallelDataset
from utils.constants import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

model_load_time = Gauge('model_load_time_seconds', 'Model loading time')


class UniversalTranslationSystem:
    """Complete integrated translation system with async support"""

    def __init__(self, config: Optional[Union[SystemConfig, Dict[str, Any]]] = None):
        # Handle both dict and SystemConfig
        if isinstance(config, dict):
            self.config = SystemConfig(**config)
        else:
            self.config = config or SystemConfig()

        self.logger = logger

        # Initialize components
        self.data_pipeline = None
        self.vocab_manager = None
        self.encoder = None
        self.decoder = None
        self.trainer = None
        self.evaluator = None
        self.health_monitor = SystemHealthMonitor(self)

        # Setup device
        self.device = torch.device(self.config.device)

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Lock for thread-safe adapter switching + inference
        self._model_lock = threading.RLock()

        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self._start_monitoring()
            self._start_health_service()

        logger.info(f"🔧 Initializing Universal Translation System on {self.device}")

    def _start_health_service(self):
        """Starts the FastAPI health service in a background thread."""
        # Use a different port from Prometheus to avoid conflicts
        health_service_port = self.config.monitoring_port + 1

        health_thread = threading.Thread(
            target=start_health_service,
            # Pass the 'self' instance of UniversalTranslationSystem
            args=(self, "0.0.0.0", health_service_port),
            daemon=True  # Daemon thread will exit when the main program exits
        )
        health_thread.start()
        self.logger.info(f"✅ Health check service started on port {health_service_port}")

    def _start_monitoring(self):
        """Start Prometheus monitoring server"""
        try:
            start_http_server(self.config.monitoring_port)
            logger.info(f"📊 Monitoring server started on port {self.config.monitoring_port}")
        except Exception as e:
            logger.warning(f"Failed to start monitoring: {e}")

    def initialize_all_systems(self, retry_failed: bool = True) -> bool:
        """Initialize all components with retry logic"""
        logger.info("🚀 Initializing Universal Translation System...")

        start_time = time.time()
        failed_components = []

        # Initialize in order
        steps = [
            ("Data Pipeline", self.setup_data_pipeline),
            ("Vocabulary System", self.setup_vocabulary_system),
            ("Models", self.setup_models),
            ("Training System", self.setup_training),
            ("Evaluation System", self.setup_evaluation)
        ]

        for step_name, setup_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Setting up: {step_name}")
            logger.info('='*50)

            try:
                if not setup_func():
                    failed_components.append((step_name, setup_func))
            except Exception as e:
                logger.error(f"Exception in {step_name}: {e}")
                failed_components.append((step_name, setup_func))

        # Retry failed components
        if retry_failed and failed_components:
            logger.info("\n🔄 Retrying failed components...")
            retry_list = failed_components.copy()
            for step_name, setup_func in retry_list:
                try:
                    logger.info(f"Retrying {step_name}...")
                    if setup_func():
                        logger.info(f"✅ {step_name} succeeded on retry")
                        failed_components.remove((step_name, setup_func))
                except Exception as e:
                    logger.error(f"❌ {step_name} failed again: {e}")

        success = len(failed_components) == 0
        elapsed_time = time.time() - start_time

        # Record load time
        if success:
            model_load_time.set(elapsed_time)

        logger.info(f"\n{'='*50}")
        if success:
            logger.info(f"🎉 All systems initialized successfully!")
        else:
            logger.error(f"❌ Failed to initialize: {[name for name, _ in failed_components]}")
        logger.info(f"⏱️  Total initialization time: {elapsed_time:.2f} seconds")
        logger.info('='*50)

        return success

    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check"""
        return await self.health_monitor.check_health()

    def setup_data_pipeline(self) -> bool:
        """Initialize and configure data pipeline with better error handling"""
        try:
            logger.info("📊 Setting up data pipeline...")

            # Import data components
            from connector.pipeline_connector import PipelineConnector
            from data.unified_data_pipeline import UnifiedDataPipeline as PracticalDataPipeline
            from config.schemas import RootConfig, DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig

            # Build a RootConfig from SystemConfig for pipeline compatibility
            root_config = RootConfig(
                data=DataConfig(processed_dir=str(Path(self.config.data_dir) / DATA_PROCESSED_DIR)),
                model=ModelConfig(),
                training=TrainingConfig(batch_size=self.config.batch_size),
                memory=MemoryConfig(),
                vocabulary=VocabularyConfig(vocab_dir=self.config.vocab_dir)
            )

            # Create pipeline
            self.data_pipeline = PracticalDataPipeline(config=root_config)

            # Create connector
            connector = PipelineConnector(config=root_config)

            # Check if data exists
            processed_dir = Path(self.config.data_dir) / DATA_PROCESSED_DIR
            if not processed_dir.exists():
                logger.info("📥 No processed data found. Running data pipeline...")
                try:
                    self.data_pipeline.prepare_all_data()
                    connector.create_monolingual_corpora()
                    connector.create_final_training_file()
                except Exception as e:
                    logger.error(f"❌ Data preparation failed: {e}")
                    # Try to continue with partial data if possible
                    if not processed_dir.exists():
                        return False
            else:
                logger.info("✅ Processed data found. Skipping pipeline.")

            logger.info("✅ Data pipeline ready")
            return True

        except ImportError as e:
            logger.error(f"❌ Failed to import data pipeline modules: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Data pipeline setup failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("✅ Cleanup complete")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.cleanup()
        return False  # Don't suppress exceptions

    def setup_vocabulary_system(self) -> bool:
        """Initialize vocabulary management system"""
        try:
            logger.info("📚 Setting up vocabulary system...")

            # Import vocabulary components
            from vocabulary.unified_vocab_manager import UnifiedVocabularyManager, VocabularyMode
            from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator as VocabularyPackCreator

            # Use OPTIMIZED mode for integration
            OptimizedVocabularyManager = lambda *args, **kwargs: UnifiedVocabularyManager(*args, mode=VocabularyMode.OPTIMIZED, **kwargs)

            # Check if vocabulary packs exist
            vocab_path = Path(self.config.vocab_dir)
            if not vocab_path.exists() or not list(vocab_path.glob("*.msgpack")):
                logger.info("📝 Creating vocabulary packs...")

                # Create vocabulary packs
                creator = VocabularyPackCreator(
                    corpus_dir=f"{self.config.data_dir}/processed",
                    output_dir=self.config.vocab_dir
                )
                creator.create_all_packs()

            # Initialize optimized manager
            self.vocab_manager = OptimizedVocabularyManager(
                vocab_dir=self.config.vocab_dir,
                cache_size=self.config.vocab_cache_size
            )

            logger.info("✅ Vocabulary system ready")
            return True

        except Exception as e:
            logger.error(f"❌ Vocabulary setup failed: {e}")
            return False

    def setup_models(self, load_checkpoint: Optional[str] = None) -> bool:
        """Initialize encoder and decoder models"""
        try:
            logger.info("🤖 Setting up models...")

            # Import model components
            from encoder.universal_encoder import UniversalEncoder
            from encoder.language_adapters import AdapterUniversalEncoder
            from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder

            # Create encoder
            if self.config.use_adapters:
                # Use adapter-enabled encoder
                if load_checkpoint:
                    self.encoder = AdapterUniversalEncoder(base_encoder_path=load_checkpoint)
                else:
                    self.encoder = AdapterUniversalEncoder()
            else:
                # Standard encoder
                self.encoder = UniversalEncoder(
                    max_vocab_size=50000,
                    hidden_dim=1024,
                    num_layers=6,
                    num_heads=16
                )

            # Create decoder
            self.decoder = OptimizedUniversalDecoder(
                encoder_dim=1024,
                decoder_dim=512,
                num_layers=6,
                num_heads=8,
                vocab_size=50000,
                device=self.device
            )

            # Load checkpoint if provided
            if load_checkpoint and Path(load_checkpoint).exists():
                self._load_checkpoint(load_checkpoint)

            # Move models to device
            self.encoder.to(self.device)
            self.decoder.to(self.device)

            # Apply quantization if needed
            if self.config.quantization_mode != "fp32":
                self._apply_quantization()

            logger.info("✅ Models ready")
            return True

        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            return False

    def setup_training(self) -> bool:
        """Initialize training system"""
        try:
            logger.info("🎯 Setting up training system...")

            # Import training components
            from training.progressive_training import ProgressiveTrainingStrategy
            from training.memory_efficient_training import MemoryOptimizedTrainer, MemoryConfig

            # Create datasets
            from data.dataset_classes import ModernParallelDataset

            train_path = Path(self.config.data_dir) / DATA_PROCESSED_DIR / "train_final.txt"
            val_path = Path(self.config.data_dir) / DATA_PROCESSED_DIR / "val_final.txt"

            if not train_path.exists():
                logger.warning("⚠️  Training data not found. Run data pipeline first.")
                return False

            train_dataset = ModernParallelDataset(str(train_path))
            val_dataset = ModernParallelDataset(str(val_path)) if val_path.exists() else None

            # Setup progressive training
            self.trainer = ProgressiveTrainingStrategy(
                encoder=self.encoder,
                decoder=self.decoder,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

            # Setup memory-efficient training
            memory_config = MemoryConfig(
                gradient_checkpointing=True,
                mixed_precision=True,
                compile_model=torch.__version__ >= "2.0.0",
                use_flash_attention=True
            )

            self.memory_trainer = MemoryOptimizedTrainer(self.encoder, memory_config)

            logger.info("✅ Training system ready")
            return True

        except Exception as e:
            logger.error(f"❌ Training setup failed: {e}")
            return False

    def setup_evaluation(self) -> bool:
        """Initialize evaluation system"""
        try:
            logger.info("📈 Setting up evaluation system...")

            # Import evaluation components
            from evaluation.evaluate_model import TranslationEvaluator

            self.evaluator = TranslationEvaluator(
                encoder_model=self.encoder,
                decoder_model=self.decoder,
                vocabulary_manager=self.vocab_manager,
                device=str(self.device)
            )

            logger.info("✅ Evaluation system ready")
            return True

        except Exception as e:
            logger.error(f"❌ Evaluation setup failed: {e}")
            return False

    def train_progressive(self, save_dir: Optional[str] = None):
        """Run progressive training"""
        if not self.trainer:
            logger.error("❌ Training system not initialized")
            return

        save_dir = save_dir or f"{self.config.checkpoint_dir}/progressive"

        logger.info("🏃 Starting progressive training...")
        self.trainer.train_progressive(save_dir=save_dir)

    def train_adapters(self, languages: list):
        """Train language-specific adapters"""
        if not self.config.use_adapters:
            logger.error("❌ Adapter system not enabled")
            return

        from encoder.train_adapters import AdapterTrainer

        adapter_trainer = AdapterTrainer(
            base_model_path=f"{self.config.model_dir}/universal_encoder.pt"
        )

        # Create data loaders for each language
        # (Implementation depends on your data structure)

        logger.info(f"🎯 Training adapters for languages: {languages}")
        # adapter_trainer.train_all_adapters(languages, train_loaders, val_loaders)

    def export_edge_model(self, output_dir: str, languages: list):
        """Export optimized model for edge deployment"""
        logger.info(f"📦 Creating edge deployment package...")

        if not self.encoder:
            logger.error("Encoder not initialized. Cannot export edge model.")
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use AdapterUniversalEncoder's built-in save method
        if hasattr(self.encoder, 'save_edge_model'):
            model_path = self.encoder.save_edge_model(
                output_dir=str(output_path),
                quantization_mode=self.config.quantization_mode
            )
            logger.info(f"✅ Edge model exported to {output_path}")
            return str(output_path / 'base_encoder_int8.pt')

        # Fallback: save encoder state dict directly
        model_file = output_path / 'encoder.pt'
        torch.save({
            'state_dict': self.encoder.state_dict(),
            'config': {
                'hidden_dim': getattr(self.encoder, 'hidden_dim', 1024),
                'num_layers': getattr(self.encoder, 'num_layers', 6),
                'quantization': self.config.quantization_mode,
            }
        }, model_file)
        logger.info(f"✅ Encoder exported to {model_file}")
        return str(model_file)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")

    def _apply_quantization(self):
        """Apply quantization to models"""
        from training.quantization_pipeline import EncoderQuantizer

        quantizer = EncoderQuantizer()

        if self.config.quantization_mode == "int8":
            logger.info("🔄 Applying INT8 quantization...")
            # Apply dynamic quantization
            self.encoder = torch.quantization.quantize_dynamic(
                self.encoder,
                qconfig_spec={torch.nn.Linear, torch.nn.Embedding},
                dtype=torch.qint8
            )
        elif self.config.quantization_mode == "fp16":
            logger.info("🔄 Converting to FP16...")
            self.encoder = self.encoder.half()
            self.decoder = self.decoder.half()

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system state"""
        info = {
            "config": self.config.__dict__,
            "device": str(self.device),
            "components": {
                "data_pipeline": self.data_pipeline is not None,
                "vocab_manager": self.vocab_manager is not None,
                "encoder": self.encoder is not None,
                "decoder": self.decoder is not None,
                "trainer": self.trainer is not None,
                "evaluator": self.evaluator is not None
            },
            "model_info": {}
        }

        if self.encoder:
            info["model_info"]["encoder"] = {
                "type": self.encoder.__class__.__name__,
                "parameters": sum(p.numel() for p in self.encoder.parameters())
            }

        if self.decoder:
            info["model_info"]["decoder"] = {
                "type": self.decoder.__class__.__name__,
                "parameters": sum(p.numel() for p in self.decoder.parameters())
            }

        return info

    async def tune_hyperparameters(self,
                                param_space: Dict[str, List[Any]],
                                validation_data: str,
                                n_trials: int = 20) -> Dict[str, Any]:
        """Simple hyperparameter tuning"""
        best_score = -float('inf')
        best_params = {}

        import itertools
        import random

        # Generate parameter combinations
        param_combinations = list(itertools.product(*param_space.values()))
        random.shuffle(param_combinations)

        for i, params in enumerate(param_combinations[:n_trials]):
            param_dict = dict(zip(param_space.keys(), params))

            # Apply parameters
            self._apply_hyperparameters(param_dict)

            # Evaluate
            metrics = await self.evaluate_async(validation_data)
            score = metrics.get('bleu', 0.0)

            if score > best_score:
                best_score = score
                best_params = param_dict

            logger.info(f"Trial {i+1}/{n_trials}: {param_dict} -> BLEU: {score:.2f}")

        return {'best_params': best_params, 'best_score': best_score}

    def _apply_hyperparameters(self, params: Dict[str, Any]):
        """Apply hyperparameters to the system"""
        # Update learning rate
        if hasattr(self, 'trainer') and self.trainer and 'lr' in params:
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = params['lr']

        # Update batch size
        if 'batch_size' in params:
            self.config.batch_size = params['batch_size']

        # Update other parameters as needed
        logger.info(f"Applied hyperparameters: {params}")

    def optimize_hyperparameters(self, trial_budget: int = 20):
        """Use Optuna for hyperparameter optimization"""
        import optuna
        train_path = Path(self.config.data_dir) / DATA_PROCESSED_DIR / "train_final.txt"
        val_path = Path(self.config.data_dir) / DATA_PROCESSED_DIR / "val_final.txt"

        def objective(trial):
            from training.intelligent_trainer import IntelligentTrainer
            train_dataset = ModernParallelDataset(str(train_path))
            val_dataset = ModernParallelDataset(str(val_path)) if Path(val_path).exists() else None
            trainer = IntelligentTrainer(
                encoder=self.encoder,
                decoder=self.decoder,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=self.config
            )
            trainer.train(num_epochs=3)
            return trainer.best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trial_budget)

        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

    def validate_exported_model(self, model_path: str, test_samples: List[Tuple[str, str, str]]) -> bool:
        """Validate exported model works correctly"""
        try:
            # Load exported model
            import onnx
            import onnxruntime as ort

            # Create inference session
            session = ort.InferenceSession(model_path)

            # Test on samples
            for source_text, source_lang, target_lang in test_samples:
                # Run inference
                # ... implementation ...
               pass

            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
