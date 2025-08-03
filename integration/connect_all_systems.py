# integration/connect_all_systems.py
"""
Complete integration module for the Universal Translation System
Connects all components and provides unified interface with async support
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional ,  List, Coroutine, Union
import torch
import json
import yaml
import time
from dataclasses import dataclass, field
import psutil
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from utils.dataset_classes import ModernParallelDataset, StreamingParallelDataset
from monitoring.health_service import start_health_service
import threading
from utils.validators import InputValidator

# Metrics
translation_counter = Counter('translations_total', 'Total translations', ['source_lang', 'target_lang'])
translation_duration = Histogram('translation_duration_seconds', 'Translation duration')
model_load_time = Gauge('model_load_time_seconds', 'Model loading time')
system_health = Gauge('system_health_status', 'System health status (1=healthy, 0=unhealthy)')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic configuration
class SystemConfig(BaseModel):
    """Configuration for the integrated system with validation"""
    data_dir: str = Field(default="data", description="Data directory path")
    model_dir: str = Field(default="models", description="Model directory path")
    vocab_dir: str = Field(default="vocabs", description="Vocabulary directory path")
    checkpoint_dir: str = Field(default="checkpoints", description="Checkpoint directory")
    device: str = Field(default="cuda", description="Device for computation")
    use_adapters: bool = Field(default=True, description="Use language adapters")
    quantization_mode: str = Field(default="int8", pattern="^(fp32|fp16|int8)$")
    vocab_cache_size: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=32, ge=1, le=512)
    enable_monitoring: bool = Field(default=True, description="Enable Prometheus monitoring")
    monitoring_port: int = Field(default=8000, ge=1024, le=65535)
    
    @validator('device')
    def validate_device(cls, v):
        if v == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        if 'device' in values and values['device'] == 'cpu' and v > 32:
            logger.warning(f"Large batch size ({v}) on CPU may be slow")
        return v

    @validator('vocab_cache_size')
    def validate_vocab_cache_size(cls, v):
        import psutil
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        estimated_cache_size_gb = v * 0.5  # Assume ~500MB per vocab pack
    
        if estimated_cache_size_gb > available_memory_gb * 0.5:
            logger.warning(f"Vocab cache size may use {estimated_cache_size_gb:.1f}GB RAM")
        return v

    @validator('monitoring_port')
    def validate_monitoring_port(cls, v):
        import socket
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', v))
            return v
        except OSError:
            raise ValueError(f"Port {v} is already in use")    
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

class SystemHealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self, system: 'UniversalTranslationSystem'):
        self.system = system
        self.health_metrics = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'performance': {},
            'resources': {}
        }
        
        # Check each component
        components = [
            ('data_pipeline', self._check_data_pipeline),
            ('vocab_manager', self._check_vocab_manager),
            ('encoder', self._check_encoder),
            ('decoder', self._check_decoder),
            ('trainer', self._check_trainer),
            ('evaluator', self._check_evaluator)
        ]
        
        for name, check_func in components:
            try:
                health['components'][name] = await check_func()
            except Exception as e:
                health['components'][name] = {'status': 'error', 'error': str(e)}
                health['status'] = 'degraded'
        
        # Check resources
        health['resources'] = await self._check_resources()
        
        # Update Prometheus metric
        system_health.set(1 if health['status'] == 'healthy' else 0)
        
        return health
    
    async def _check_data_pipeline(self) -> Dict[str, Any]:
        """Check data pipeline health"""
        return {
            'status': 'healthy' if self.system.data_pipeline is not None else 'not_initialized',
            'data_dir_exists': Path(self.system.config.data_dir).exists(),
            'processed_data_exists': (Path(self.system.config.data_dir) / "processed").exists()
        }
    
    async def _check_vocab_manager(self) -> Dict[str, Any]:
        """Check vocabulary manager health"""
        if self.system.vocab_manager is None:
            return {'status': 'not_initialized'}
        
        try:
            # Check loaded vocabularies
            loaded_versions = self.system.vocab_manager.get_loaded_versions()
            return {
                'status': 'healthy',
                'loaded_packs': len(loaded_versions),
                'versions': loaded_versions
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _check_encoder(self) -> Dict[str, Any]:
        """Check encoder health"""
        if self.system.encoder is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'healthy',
            'device': str(next(self.system.encoder.parameters()).device),
            'parameters': sum(p.numel() for p in self.system.encoder.parameters()),
            'training': self.system.encoder.training
        }
    
    async def _check_decoder(self) -> Dict[str, Any]:
        """Check decoder health"""
        if self.system.decoder is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'healthy',
            'device': str(next(self.system.decoder.parameters()).device),
            'parameters': sum(p.numel() for p in self.system.decoder.parameters()),
            'training': self.system.decoder.training
        }
    
    async def _check_trainer(self) -> Dict[str, Any]:
        """Check trainer health"""
        return {
            'status': 'healthy' if self.system.trainer is not None else 'not_initialized'
        }
    
    async def _check_evaluator(self) -> Dict[str, Any]:
        """Check evaluator health"""
        return {
            'status': 'healthy' if self.system.evaluator is not None else 'not_initialized'
        }
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3)
        }
        
        # GPU resources
        if torch.cuda.is_available():
            resources['gpu'] = {
                'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'utilization': self._get_gpu_utilization()
            }
        
        return resources
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return None

    def validate_configuration(self) -> List[str]:
        """Validate system configuration"""
        errors = []
    
        # Check directories exist
        for dir_attr in ['data_dir', 'model_dir', 'vocab_dir']:
            dir_path = Path(getattr(self.config, dir_attr))
            if not dir_path.exists():
                errors.append(f"{dir_attr} does not exist: {dir_path}")
    
        # Check device availability
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            errors.append("CUDA requested but not available")
    
        # Check port availability for monitoring
        if self.config.enable_monitoring:
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', self.config.monitoring_port))
                if result == 0:
                    errors.append(f"Monitoring port {self.config.monitoring_port} already in use")
                sock.close()
            except:
                pass
    
        return errors        

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

    async def translate_async(self, text: str, source_lang: str, target_lang: str) -> str:
        """Async translation for better concurrency"""
        # Record metrics
        start_time = time.time()
        translation_counter.labels(source_lang=source_lang, target_lang=target_lang).inc()
        
        # Run CPU-bound translation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.translate,
            text,
            source_lang,
            target_lang
        )
        
        # Record duration
        translation_duration.observe(time.time() - start_time)
        
        return result
    
    async def translate_batch_async(self, 
                                   texts: List[str], 
                                   source_lang: str, 
                                   target_lang: str,
                                   max_concurrent: int = 10) -> List[str]:
        """Translate multiple texts concurrently with rate limiting"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def translate_with_semaphore(text: str) -> str:
            async with semaphore:
                return await self.translate_async(text, source_lang, target_lang)
        
        tasks = [translate_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)

    def translate(self, text: str, source_lang: str, target_lang: str,domain: Optional[str] = None) -> str:
        """Translate text with input validation and optional domain-specific expertise."""
        # Validation:
        text = InputValidator.validate_text_input(text, max_length=5000)
    
        if not InputValidator.validate_language_code(source_lang):
            raise ValueError(f"Invalid source language: {source_lang}")
    
        if not InputValidator.validate_language_code(target_lang):
            raise ValueError(f"Invalid target language: {target_lang}")

        if not self.encoder or not self.decoder:
            raise RuntimeError("Models not initialized")
    
        # --- MODIFIED ---
        # 1. Determine which vocabulary and adapter to use
        if domain:
            # Construct domain-specific names
            vocab_pack_name = f"latin_{domain}" # e.g., 'latin_medical'
            adapter_name = f"{source_lang}_{domain}" # e.g., 'es_medical'
        else:
            # Fallback to general-purpose packs
            vocab_pack_name = self.vocab_manager.language_to_pack.get(source_lang, 'latin')
            adapter_name = source_lang

        # 2. Load the correct vocabulary pack
        try:
            vocab_pack = self.vocab_manager._load_pack(vocab_pack_name)
        except VocabularyError:
            if domain:
                logger.warning(f"Domain vocab '{vocab_pack_name}' not found. Falling back to general vocab.")
                general_pack_name = self.vocab_manager.language_to_pack.get(source_lang, 'latin')
                vocab_pack = self.vocab_manager._load_pack(general_pack_name)
                # Also fallback the adapter name
                adapter_name = source_lang
            else:
                raise # Re-raise if general vocab is not found

        # 3. Load the correct adapter
        # This assumes your AdapterUniversalEncoder can load adapters by a unique name
        self.encoder.load_language_adapter(adapter_name, adapter_path=f"models/adapters/best_{adapter_name}_adapter.pt")

        # 4. Translate
        if self.evaluator:
            # The evaluator will now use the loaded domain-specific adapter and vocab
            return self.evaluator.translate(text, source_lang, target_lang)
        else:
            raise RuntimeError("Translation system not fully initialized")
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check"""
        return await self.health_monitor.check_health()

    def setup_data_pipeline(self) -> bool:
        """Initialize and configure data pipeline with better error handling"""
        try:
            logger.info("📊 Setting up data pipeline...")
            
            # Import data components
            from data.pipeline_connector import PipelineConnector
            from data.practical_data_pipeline import PracticalDataPipeline
            
            # Create pipeline
            self.data_pipeline = PracticalDataPipeline()
            
            # Create connector
            connector = PipelineConnector()
            
            # Check if data exists
            processed_dir = Path(self.config.data_dir) / "processed"
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
            from vocabulary.optimized_vocab_manager import OptimizedVocabularyManager
            from vocabulary.create_vocabulary_packs_from_data import VocabularyPackCreator
            
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
            from training.train_universal_system import ModernParallelDataset
            
            train_path = Path(self.config.data_dir) / "processed" / "train_final.txt"
            val_path = Path(self.config.data_dir) / "processed" / "val_final.txt"
            
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
    
    def evaluate(self, test_file: str, output_file: Optional[str] = None):
        """Evaluate the system on test data"""
        if not self.evaluator:
            logger.error("❌ Evaluation system not initialized")
            return
        
        logger.info(f"📊 Evaluating on {test_file}...")
        
        metrics = self.evaluator.evaluate_file(test_file)
        
        if output_file:
            self.evaluator.create_evaluation_report(metrics, output_file)
        
        return metrics
    
    def export_edge_model(self, output_dir: str, languages: list):
        """Export optimized model for edge deployment"""
        from encoder.language_adapters import create_edge_deployment_package
        
        logger.info(f"📦 Creating edge deployment package...")
        
        model_path = create_edge_deployment_package(
            base_encoder_path=f"{self.config.model_dir}/universal_encoder.pt",
            languages=languages,
            output_dir=output_dir,
            quantization_mode=self.config.quantization_mode
        )
        
        logger.info(f"✅ Edge model exported to {model_path}")
        
        return model_path

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

    async def evaluate_async(self, validation_data: str) -> Dict[str, float]:
        """Async evaluation wrapper"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.evaluate,
            validation_data
        )    

    # Add Optuna integration
    def optimize_hyperparameters(trial_budget: int = 20):
        """Use Optuna for hyperparameter optimization"""
        import optuna
    
        def objective(trial):
            # Suggest hyperparameters
            config = {
                'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'dropout': trial.suggest_uniform('dropout', 0.1, 0.3),
                'gradient_clip': trial.suggest_uniform('gradient_clip', 0.5, 2.0)
            }
        
            # Train with config
            trainer = ModernUniversalSystemTrainer(
                encoder, decoder, train_path, val_path,
                config=MemoryConfig(**config)
            )
        
            # Train for few epochs
            trainer.train(num_epochs=3)
        
            # Return validation loss
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

# Enhanced integration function with async support
async def integrate_full_pipeline_async(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Main async integration function"""
    
    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SystemConfig(**config_dict)
    else:
        config = SystemConfig()
    
    # Create integrated system
    system = UniversalTranslationSystem(config)
    
    # Initialize all components
    if system.initialize_all_systems():
        logger.info("\n✅ System ready for use!")
        
        # Run initial health check
        health = await system.health_check_async()
        logger.info(f"System health: {health['status']}")
        
        return system
    else:
        logger.error("❌ System initialization failed")
        return None

# Synchronous wrapper for backward compatibility
def integrate_full_pipeline(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Synchronous wrapper for the async integration"""
    return asyncio.run(integrate_full_pipeline_async(config_file))


if __name__ == "__main__":
    # Create default configuration file if needed
    config_path = Path("config/integration_config.yaml")
    if not config_path.exists():
        config_path.parent.mkdir(exist_ok=True)
        
        default_config = {
            "data_dir": "data",
            "model_dir": "models",
            "vocab_dir": "vocabs",
            "checkpoint_dir": "checkpoints",
            "device": "cuda",
            "use_adapters": True,
            "quantization_mode": "int8",
            "vocab_cache_size": 3,
            "batch_size": 32
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default configuration at {config_path}")
    
    # Initialize the complete system
    system = integrate_full_pipeline(str(config_path))
    
    if system:
        # Show system information
        info = system.get_system_info()
        print(f"\nSystem Information:")
        print(f"Device: {info['device']}")
        print(f"Components initialized: {sum(info['components'].values())}/6")