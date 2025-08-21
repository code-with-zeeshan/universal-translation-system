# main.py - Unified entry point for the Universal Translation System
#!/usr/bin/env python
"""
Unified entry point for the Universal Translation System.
Combines run_system.py and run_training.py functionality.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from enum import Enum

import torch

from utils.logging_config import setup_logging
from utils.final_integration import SystemIntegrator
from integration.connect_all_systems import integrate_full_pipeline
from config.schemas import load_config

# Setup logging early
setup_logging(log_dir="logs", log_level="INFO")
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class OperationMode(Enum):
    """System operation modes"""
    SETUP = "setup"
    TRAIN = "train"
    EVALUATE = "evaluate"
    TRANSLATE = "translate"
    BENCHMARK = "benchmark"
    EXPORT = "export"


class HardwareConfig:
    """Hardware detection and configuration"""
    
    @staticmethod
    def detect_gpus() -> Tuple[int, List[str]]:
        """
        Detect available GPUs and their types.
        
        Returns:
            Tuple of (gpu_count, gpu_names)
        """
        try:
            gpu_count = torch.cuda.device_count()
            gpu_names = []
            
            if gpu_count > 0:
                for i in range(gpu_count):
                    gpu_names.append(torch.cuda.get_device_name(i))
                logger.info(f"Detected {gpu_count} GPU(s): {', '.join(gpu_names)}")
            else:
                logger.warning("No CUDA-enabled GPUs detected")
            
            return gpu_count, gpu_names
            
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
            return 0, []
    
    @staticmethod
    def get_recommended_config(gpu_count: int, gpu_names: List[str]) -> str:
        """
        Get recommended configuration file based on hardware.
        
        Args:
            gpu_count: Number of GPUs
            gpu_names: List of GPU model names
            
        Returns:
            Path to recommended config file
        """
        if gpu_count == 0:
            return "config/training_cpu.yaml"
        
        # Detect GPU type from name
        primary_gpu = gpu_names[0].lower() if gpu_names else ""
        
        # Map GPU names to config files
        gpu_configs = {
            'h100': 'config/training_h100.yaml',
            'a100': 'config/training_a100.yaml',
            'v100': 'config/training_v100.yaml',
            't4': 'config/training_t4.yaml',
            'l4': 'config/training_l4.yaml',
            'rtx 4090': 'config/training_rtx4090.yaml',
            'rtx 3090': 'config/training_rtx3090.yaml',
            'rtx 3080': 'config/training_rtx3080.yaml',
            'rtx 3060': 'config/training_rtx3060.yaml',
        }
        
        # Find matching config
        for gpu_key, config_path in gpu_configs.items():
            if gpu_key in primary_gpu:
                if gpu_count > 1:
                    # Check for multi-GPU specific config
                    multi_config = config_path.replace('.yaml', '_multi.yaml')
                    if Path(multi_config).exists():
                        return multi_config
                return config_path
        
        # Default configs
        if gpu_count == 1:
            return "config/training_v100.yaml"  # Default single GPU
        else:
            return "config/training_v100_multi.yaml"  # Default multi GPU


class UniversalTranslationSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Universal Translation System.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or "config/base.yaml"
        self.hardware = HardwareConfig()
        self.gpu_count, self.gpu_names = self.hardware.detect_gpus()
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info("Universal Translation System initialized")
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Hardware: {self.gpu_count} GPU(s)")
    
    def _load_config(self) -> Dict:
        """Load system configuration"""
        try:
            config = load_config(self.config_path)
            return config.dict()
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def validate_system(self, verbose: bool = False) -> bool:
        """
        Validate system setup and dependencies.
        
        Args:
            verbose: Print detailed validation info
            
        Returns:
            True if system is valid
        """
        logger.info("🔍 Validating system setup...")
        
        checks = {
            'pytorch': self._check_pytorch(),
            'cuda': self._check_cuda(),
            'dependencies': self._check_dependencies(),
            'data': self._check_data_availability(),
            'models': self._check_model_paths(),
            'config': self._check_config_validity()
        }
        
        if verbose:
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                logger.info(f"{status} {check.capitalize()}: {'OK' if passed else 'FAILED'}")
        
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info("✅ System validation passed")
        else:
            logger.error("❌ System validation failed")
            failed = [k for k, v in checks.items() if not v]
            logger.error(f"Failed checks: {', '.join(failed)}")
        
        return all_passed
    
    def _check_pytorch(self) -> bool:
        """Check PyTorch installation"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability (optional)"""
        try:
            import torch
            # CUDA is optional, so we just log the status
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.version.cuda}")
            else:
                logger.info("CUDA not available (CPU mode)")
            return True  # Not a failure condition
        except:
            return True
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies"""
        required = ['numpy', 'sentencepiece', 'msgpack', 'yaml', 'tqdm']
        missing = []
        
        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            logger.error(f"Missing dependencies: {missing}")
            return False
        return True
    
    def _check_data_availability(self) -> bool:
        """Check if data directories exist"""
        data_dirs = ['data/processed', 'data/raw', 'vocabs']
        missing = []
        
        for dir_path in data_dirs:
            if not Path(dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            logger.warning(f"Missing data directories: {missing}")
            # Create them
            for dir_path in missing:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        
        return True
    
    def _check_model_paths(self) -> bool:
        """Check model directories"""
        model_dirs = ['models/encoder', 'models/decoder', 'models/production']
        
        for dir_path in model_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def _check_config_validity(self) -> bool:
        """Check if configuration is valid"""
        if not Path(self.config_path).exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return False
        
        # Could add more config validation here
        return True
    
    def setup(self, force: bool = False) -> bool:
        """
        Setup and initialize the system.
        
        Args:
            force: Force re-initialization even if already setup
            
        Returns:
            True if successful
        """
        logger.info("🚀 Setting up Universal Translation System...")
        
        # Validate first
        if not self.validate_system(verbose=True):
            return False
        
        # Check if already initialized
        if not force and Path("models/production/encoder.pt").exists():
            logger.info("System already initialized. Use --force to reinitialize.")
            return True
        
        # Initialize components
        try:
            system = integrate_full_pipeline(self.config_path)
            
            if system:
                logger.info("✅ System initialized successfully!")
                return True
            else:
                logger.error("❌ System initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def train(self, 
             gpu_selection: Optional[str] = None,
             distributed: bool = None,
             config_override: Optional[str] = None) -> int:
        """
        Start training with flexible GPU configuration.
        
        Args:
            gpu_selection: GPU selection ('all', '1', '2', etc.) or None for interactive
            distributed: Force distributed training
            config_override: Override config file
            
        Returns:
            Exit code
        """
        logger.info("🎯 Starting training...")
        
        # Determine configuration
        if config_override:
            config_file = config_override
        else:
            config_file = self.hardware.get_recommended_config(
                self.gpu_count, self.gpu_names
            )
        
        # Handle GPU selection
        if self.gpu_count == 0:
            logger.warning("⚠️ No GPUs detected. Running on CPU (very slow).")
            return self._run_cpu_training(config_file)
        
        elif self.gpu_count == 1:
            logger.info("✅ Starting training on 1 GPU")
            return self._run_single_gpu_training(config_file)
        
        else:
            # Multiple GPUs available
            gpus_to_use = self._get_gpu_selection(gpu_selection)
            
            if gpus_to_use == 1 and not distributed:
                logger.info("✅ Starting training on 1 GPU")
                return self._run_single_gpu_training(config_file)
            else:
                logger.info(f"✅ Starting distributed training on {gpus_to_use} GPUs")
                return self._run_distributed_training(gpus_to_use, config_file)
    
    def _get_gpu_selection(self, selection: Optional[str] = None) -> int:
        """
        Get GPU selection interactively or from argument.
        
        Args:
            selection: Pre-specified selection or None for interactive
            
        Returns:
            Number of GPUs to use
        """
        if selection:
            if selection.lower() == 'all':
                return self.gpu_count
            try:
                num = int(selection)
                if 1 <= num <= self.gpu_count:
                    return num
                else:
                    logger.warning(f"Invalid GPU count: {num}")
            except ValueError:
                logger.warning(f"Invalid GPU selection: {selection}")
        
        # Interactive selection
        logger.info(f"\nFound {self.gpu_count} GPUs: {', '.join(self.gpu_names)}")
        
        while True:
            choice = input(
                f"How many GPUs to use? (1-{self.gpu_count}, or 'all') [all]: "
            )
            choice = choice.strip().lower() or 'all'
            
            if choice == 'all':
                return self.gpu_count
            
            try:
                num = int(choice)
                if 1 <= num <= self.gpu_count:
                    return num
                else:
                    logger.warning(f"Please enter 1-{self.gpu_count}")
            except ValueError:
                logger.warning("Please enter a number or 'all'")
    
    def _run_cpu_training(self, config_file: str) -> int:
        """Run CPU training"""
        command = [
            sys.executable,
            "training/train_universal_system.py",
            "--config", config_file,
            "--device", "cpu"
        ]
        return self._run_command(command)
    
    def _run_single_gpu_training(self, config_file: str) -> int:
        """Run single GPU training"""
        command = [
            sys.executable,
            "training/train_universal_system.py",
            "--config", config_file
        ]
        return self._run_command(command)
    
    def _run_distributed_training(self, num_gpus: int, config_file: str) -> int:
        """Run distributed training"""
        command = [
            sys.executable, "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={num_gpus}",
            "--use_env",
            "training/distributed_train.py",
            "--config", config_file
        ]
        return self._run_command(command)
    
    def _run_command(self, command: List[str]) -> int:
        """
        Run command with output streaming.
        
        Args:
            command: Command to execute
            
        Returns:
            Exit code
        """
        logger.info(f"\n🚀 Executing: {' '.join(command)}\n")
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"Command failed with code {process.returncode}")
            
            return process.returncode
            
        except Exception as e:
            logger.error(f"Failed to run command: {e}")
            return 1
    
    def evaluate(self, 
                checkpoint: Optional[str] = None,
                test_data: Optional[str] = None) -> int:
        """
        Run model evaluation.
        
        Args:
            checkpoint: Model checkpoint path
            test_data: Test data path
            
        Returns:
            Exit code
        """
        logger.info("📊 Starting evaluation...")
        
        try:
            from evaluation.evaluate_model import main as evaluate_main
            
            # Prepare arguments
            eval_args = {
                'config': self.config_path,
                'checkpoint': checkpoint or 'models/production/best_model.pt',
                'test_data': test_data or 'data/evaluation'
            }
            
            # Run evaluation
            result = evaluate_main(**eval_args)
            return 0 if result else 1
            
        except ImportError as e:
            logger.error(f"Evaluation module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 1
    
    def translate(self,
                 text: Optional[str] = None,
                 source_lang: str = 'en',
                 target_lang: str = 'es',
                 serve: bool = False) -> int:
        """
        Run translation (interactive or service mode).
        
        Args:
            text: Text to translate (None for interactive)
            source_lang: Source language code
            target_lang: Target language code
            serve: Start translation service
            
        Returns:
            Exit code
        """
        logger.info("🌐 Starting translation...")
        
        if serve:
            # Start translation service
            logger.info("Starting translation service...")
            try:
                from cloud_decoder.optimized_decoder import start_service
                start_service(self.config_path)
                return 0
            except ImportError:
                logger.error("Translation service not available")
                return 1
        else:
            # Interactive translation
            try:
                from encoder.universal_encoder import UniversalEncoder
                
                # Load model
                model_path = Path("models/production/encoder.pt")
                if not model_path.exists():
                    logger.error(f"Model not found: {model_path}")
                    logger.info("Please train a model first: python main.py --mode train")
                    return 1
                
                encoder = UniversalEncoder.load(model_path)
                
                if text:
                    # Single translation
                    result = encoder.translate(text, source_lang, target_lang)
                    print(f"\nTranslation: {result}")
                else:
                    # Interactive mode
                    logger.info("Interactive translation (Ctrl+C to exit)")
                    while True:
                        try:
                            text = input("\nEnter text: ")
                            if not text:
                                continue
                            
                            src = input(f"Source language [{source_lang}]: ") or source_lang
                            tgt = input(f"Target language [{target_lang}]: ") or target_lang
                            
                            result = encoder.translate(text, src, tgt)
                            print(f"Translation: {result}")
                            
                        except KeyboardInterrupt:
                            logger.info("\nExiting...")
                            break
                
                return 0
                
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                return 1
    
    def benchmark(self, num_samples: int = 1000) -> int:
        """
        Run performance benchmarks.
        
        Args:
            num_samples: Number of samples to benchmark
            
        Returns:
            Exit code
        """
        logger.info("⚡ Running benchmarks...")
        
        try:
            import time
            import numpy as np
            from encoder.universal_encoder import UniversalEncoder
            
            # Load model
            model_path = Path("models/production/encoder.pt")
            if not model_path.exists():
                logger.error("No model found for benchmarking")
                return 1
            
            encoder = UniversalEncoder.load(model_path)
            
            # Prepare test samples
            test_texts = [
                f"This is test sentence number {i} for benchmarking."
                for i in range(num_samples)
            ]
            
            # Benchmark encoding
            logger.info(f"Benchmarking {num_samples} samples...")
            
            times = []
            for text in test_texts:
                start = time.time()
                _ = encoder.encode(text, 'en')
                times.append(time.time() - start)
            
            # Calculate statistics
            times = np.array(times)
            stats = {
                'mean': np.mean(times) * 1000,  # Convert to ms
                'median': np.median(times) * 1000,
                'std': np.std(times) * 1000,
                'min': np.min(times) * 1000,
                'max': np.max(times) * 1000,
                'throughput': num_samples / np.sum(times)  # samples/sec
            }
            
            # Display results
            logger.info("\n📊 Benchmark Results:")
            logger.info(f"Samples: {num_samples}")
            logger.info(f"Mean latency: {stats['mean']:.2f}ms")
            logger.info(f"Median latency: {stats['median']:.2f}ms")
            logger.info(f"Std deviation: {stats['std']:.2f}ms")
            logger.info(f"Min latency: {stats['min']:.2f}ms")
            logger.info(f"Max latency: {stats['max']:.2f}ms")
            logger.info(f"Throughput: {stats['throughput']:.1f} samples/sec")
            
            # Save results
            import json
            with open("benchmark_results.json", 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"\nResults saved to benchmark_results.json")
            
            return 0
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return 1
    
    def export(self, 
              format: str = 'onnx',
              output_dir: str = 'models/export') -> int:
        """
        Export model for deployment.
        
        Args:
            format: Export format (onnx, torchscript, tflite)
            output_dir: Output directory
            
        Returns:
            Exit code
        """
        logger.info(f"📦 Exporting model to {format}...")
        
        try:
            from training.convert_models import ModelConverter
            
            converter = ModelConverter()
            
            # Load model
            model_path = Path("models/production/encoder.pt")
            if not model_path.exists():
                logger.error("No model found to export")
                return 1
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == 'onnx':
                output_path = Path(output_dir) / "model.onnx"
                converter.to_onnx(model_path, output_path)
            elif format == 'torchscript':
                output_path = Path(output_dir) / "model.pt"
                converter.to_torchscript(model_path, output_path)
            elif format == 'tflite':
                output_path = Path(output_dir) / "model.tflite"
                converter.to_tflite(model_path, output_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return 1
            
            logger.info(f"✅ Model exported to {output_path}")
            return 0
            
        except ImportError:
            logger.error("Model conversion module not available")
            return 1
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Universal Translation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup system
  python main.py --mode setup
  
  # Train with automatic GPU detection
  python main.py --mode train
  
  # Train with specific GPUs
  python main.py --mode train --gpus 2
  
  # Evaluate model
  python main.py --mode evaluate
  
  # Interactive translation
  python main.py --mode translate
  
  # Start translation service
  python main.py --mode translate --serve
  
  # Run benchmarks
  python main.py --mode benchmark
  
  # Export model
  python main.py --mode export --format onnx
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=[m.value for m in OperationMode],
        default='setup',
        help='Operation mode'
    )
    
    # Common arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/base.yaml',
        help='Configuration file'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate system setup'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # Training arguments
    parser.add_argument(
        '--gpus',
        type=str,
        help='GPU selection (e.g., "all", "1", "2")'
    )
    
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Force distributed training'
    )
    
    parser.add_argument(
        '--training-config',
        type=str,
        help='Override training configuration'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Model checkpoint for evaluation'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Test data path'
    )
    
    # Translation arguments
    parser.add_argument(
        '--text',
        type=str,
        help='Text to translate'
    )
    
    parser.add_argument(
        '--source-lang',
        type=str,
        default='en',
        help='Source language'
    )
    
    parser.add_argument(
        '--target-lang',
        type=str,
        default='es',
        help='Target language'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start translation service'
    )
    
    # Benchmark arguments
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for benchmark'
    )
    
    # Export arguments
    parser.add_argument(
        '--format',
        type=str,
        choices=['onnx', 'torchscript', 'tflite'],
        default='onnx',
        help='Export format'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/export',
        help='Export output directory'
    )
    
    # Setup arguments
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-initialization'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system
    system = UniversalTranslationSystem(args.config)
    
    # Validate only mode
    if args.validate_only:
        valid = system.validate_system(verbose=True)
        return 0 if valid else 1
    
    # Execute based on mode
    mode = OperationMode(args.mode)
    
    if mode == OperationMode.SETUP:
        success = system.setup(force=args.force)
        return 0 if success else 1
    
    elif mode == OperationMode.TRAIN:
        return system.train(
            gpu_selection=args.gpus,
            distributed=args.distributed,
            config_override=args.training_config
        )
    
    elif mode == OperationMode.EVALUATE:
        return system.evaluate(
            checkpoint=args.checkpoint,
            test_data=args.test_data
        )
    
    elif mode == OperationMode.TRANSLATE:
        return system.translate(
            text=args.text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            serve=args.serve
        )
    
    elif mode == OperationMode.BENCHMARK:
        return system.benchmark(num_samples=args.num_samples)
    
    elif mode == OperationMode.EXPORT:
        return system.export(
            format=args.format,
            output_dir=args.output_dir
        )
    
    else:
        logger.error(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())