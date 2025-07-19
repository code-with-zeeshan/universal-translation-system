# integration/connect_all_systems.py
"""
Complete integration module for the Universal Translation System
Connects all components and provides unified interface
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import yaml
import time
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for the integrated system"""
    data_dir: str = "data"
    model_dir: str = "models"
    vocab_dir: str = "vocabs"
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    use_adapters: bool = True
    quantization_mode: str = "int8"  # Options: fp32, fp16, int8
    vocab_cache_size: int = 3
    batch_size: int = 32


class UniversalTranslationSystem:
    """Complete integrated translation system"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.logger = logger
        
        # Initialize components
        self.data_pipeline = None
        self.vocab_manager = None
        self.encoder = None
        self.decoder = None
        self.trainer = None
        self.evaluator = None
        
        # Setup device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        logger.info(f"🔧 Initializing Universal Translation System on {self.device}")
    
    def setup_data_pipeline(self) -> bool:
        """Initialize and configure data pipeline"""
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
                self.data_pipeline.prepare_all_data()
                
                # Create monolingual corpora
                connector.create_monolingual_corpora()
                
                # Create final training file
                connector.create_final_training_file()
            else:
                logger.info("✅ Processed data found. Skipping pipeline.")
            
            logger.info("✅ Data pipeline ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data pipeline setup failed: {e}")
            return False
    
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
    
    def initialize_all_systems(self) -> bool:
        """Initialize all components of the translation system"""
        logger.info("🚀 Initializing Universal Translation System...")
        
        start_time = time.time()
        
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
            
            if not setup_func():
                logger.error(f"❌ Failed to setup {step_name}")
                return False
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🎉 All systems initialized successfully!")
        logger.info(f"⏱️  Total initialization time: {elapsed_time:.2f} seconds")
        logger.info('='*50)
        
        return True
    
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
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the integrated system"""
        if not all([self.encoder, self.decoder, self.vocab_manager]):
            logger.error("❌ System not fully initialized")
            return ""
        
        # Use evaluator's translate method
        if self.evaluator:
            return self.evaluator.translate(text, source_lang, target_lang)
        else:
            logger.error("❌ Evaluator not initialized")
            return ""
    
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


def integrate_full_pipeline(config_file: Optional[str] = None) -> UniversalTranslationSystem:
    """Main integration function"""
    
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
        
        # Print usage instructions
        print("\n" + "="*60)
        print("UNIVERSAL TRANSLATION SYSTEM READY")
        print("="*60)
        print("\nUsage examples:")
        print("1. Train progressive: system.train_progressive()")
        print("2. Evaluate: system.evaluate('test_data.tsv')")
        print("3. Translate: system.translate('Hello', 'en', 'es')")
        print("4. Export edge model: system.export_edge_model('edge_model/', ['en', 'es', 'fr'])")
        print("\nSystem info: system.get_system_info()")
        print("="*60)
        
        return system
    else:
        logger.error("❌ System initialization failed")
        return None


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