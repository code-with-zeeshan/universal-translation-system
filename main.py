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
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from enum import Enum

import torch

from utils.logging_config import setup_logging
from integration.system import UniversalTranslationSystem as IntegrationSystem
from integration.system_config import SystemConfig as IntegrationConfig
from integration.translation_api import integrate_full_pipeline
from config.schemas import load_config
from utils.constants import LOG_DIR, MODELS_PRODUCTION_DIR, ENCODER_MODEL_FILENAME, BENCHMARK_RESULTS_FILENAME, CONFIG_DIR, BASE_CONFIG_FILENAME

# Setup centralized logging early
setup_logging(log_dir=LOG_DIR, log_level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("system")

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
            return "config/archived_gpu_configs/training_cpu.yaml"
        
        # Detect GPU type from name
        primary_gpu = gpu_names[0].lower() if gpu_names else ""
        
        # Map GPU names to archived config files
        base = "config/archived_gpu_configs"
        gpu_configs = {
            'h100': f'{base}/training_h100.yaml',
            'a100': f'{base}/training_a100.yaml',
            'v100': f'{base}/training_v100.yaml',
            't4': f'{base}/training_t4.yaml',
            'l4': f'{base}/training_l4.yaml',
            'rtx 4090': f'{base}/training_rtx4090.yaml',
            'rtx 3090': f'{base}/training_rtx3090.yaml',
            'rtx 3080': f'{base}/training_rtx3080.yaml',
            'rtx 3060': f'{base}/training_rtx3060.yaml',
            'amd mi250': f'{base}/training_amd_mi250.yaml',
            'colab free': f'{base}/training_colab_free.yaml',
        }
        
        # Find matching config
        for gpu_key, config_path in gpu_configs.items():
            if gpu_key in primary_gpu:
                if gpu_count > 1:
                    # Check for multi-GPU specific archived config
                    multi_config = config_path.replace('.yaml', '_multi.yaml')
                    if Path(multi_config).exists():
                        return multi_config
                return config_path
        
        # Default configs - use archived generic configs
        if gpu_count == 1:
            return f"{base}/training_generic_gpu.yaml"  # Default single GPU
        else:
            return f"{base}/training_generic_multi_gpu.yaml"  # Default multi GPU

    @staticmethod
    def get_compile_recommendation(gpu_names: List[str]) -> Tuple[bool, str]:
        """
        Recommend torch.compile usage and mode based on GPU class.
        Returns: (compile_model, compile_mode)
        """
        primary = (gpu_names[0].lower() if gpu_names else "")
        # Map families to (mode, enabled)
        mapping = [
            (['h100', 'a100', 'amd mi250', 'mi300'], ("max-autotune", True)),
            (['v100', 'l4'], ("reduce-overhead", True)),
            (['t4', 'rtx 4090', 'rtx 4080', 'rtx 4070', 'rtx 3090', 'rtx 3080', 'rtx 3070', 'rtx 3060'], ("reduce-overhead", True)),
        ]
        for keys, (mode, enabled) in mapping:
            for k in keys:
                if k in primary:
                    return enabled, mode
        # CPU or unknown GPUs: prefer no compile or conservative default
        return False, "default"


class UniversalTranslationSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Universal Translation System.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or f"{CONFIG_DIR}/{BASE_CONFIG_FILENAME}"
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
            _ = torch.tensor(0)
            return True
        except Exception:
            return False
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability (optional)"""
        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.version.cuda}")
            else:
                logger.info("CUDA not available (CPU mode)")
            return True
        except Exception:
            return True
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies using the dependency checker script"""
        try:
            # Try to import and use the dependency checker
            sys.path.insert(0, str(project_root / "scripts"))
            try:
                import check_dependencies
                return check_dependencies.check_dependencies()
            except ImportError:
                logger.warning("Dependency checker not found, falling back to basic check")
                
                # Fallback to basic dependency checking
                required = ['numpy', 'sentencepiece', 'msgpack', 'pyyaml', 'tqdm']
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
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False
    
    def _check_data_availability(self) -> bool:
        """Check if data directories exist"""
        data_dirs = ['data/processed', 'data/raw', 'vocabulary']
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
        if not force and Path(f"{MODELS_PRODUCTION_DIR}/{ENCODER_MODEL_FILENAME}").exists():
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
        
        # Determine configuration source: dynamic by default, YAML optional
        if config_override:
            config_source = config_override
        else:
            # Prefer dynamic config generation for robustness
            config_source = "dynamic"
        
        # Handle GPU selection
        if self.gpu_count == 0:
            logger.warning("⚠️ No GPUs detected. Running on CPU (very slow).")
            return self._run_cpu_training(config_source)
        
        elif self.gpu_count == 1:
            logger.info("✅ Starting training on 1 GPU")
            return self._run_single_gpu_training(config_source)
        
        else:
            # Multiple GPUs available
            gpus_to_use = self._get_gpu_selection(gpu_selection)
            
            if gpus_to_use == 1 and not distributed:
                logger.info("✅ Starting training on 1 GPU")
                return self._run_single_gpu_training(config_source)
            else:
                logger.info(f"✅ Starting distributed training on {gpus_to_use} GPUs")
                return self._run_distributed_training(gpus_to_use, config_source)
    
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
        
        # Non-interactive default in CI/containers: use all GPUs unless overridden
        if os.environ.get("CI") or os.environ.get("NON_INTERACTIVE"):  # CI or explicit non-interactive
            return self.gpu_count
        
        # Interactive selection for local runs
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
    
    def _run_cpu_training(self, config_source: str) -> int:
        """Run CPU training"""
        command = [
            sys.executable,
            "training/intelligent_trainer.py",
            "--config", config_source,
            "--device", "cpu",
        ]
        if config_source == "dynamic":
            command.append("--dynamic")
        return self._run_command(command)
    
    def _run_single_gpu_training(self, config_source: str) -> int:
        """Run single GPU training"""
        command = [
            sys.executable,
            "training/intelligent_trainer.py",
            "--config", config_source,
        ]
        if config_source == "dynamic":
            command.append("--dynamic")
        return self._run_command(command)
    
    def _run_distributed_training(self, num_gpus: int, config_source: str) -> int:
        """Run distributed training"""
        command = [
            sys.executable, "-m",
            "torch.distributed.launch",
            f"--nproc_per_node={num_gpus}",
            "--use_env",
            "training/launch.py",
            "train",
            "--config", (config_source if config_source else "config/base.yaml"),
            "--distributed",
        ]
        if config_source == "dynamic":
            command.append("--dynamic")
        return self._run_command(command)
    
    def _run_command(self, command: List[str]) -> int:
        """
        Run command with output streaming and enhanced error handling.
        
        Args:
            command: Command to execute
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        logger.info(f"\n🚀 Executing: {' '.join(command)}\n")
        
        try:
            # Create a log file for the command output
            log_dir = Path(LOG_DIR)
            log_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = log_dir / f"command_{timestamp}.log"
            
            with open(log_file, "w") as log_fh:
                # Log the command
                log_fh.write(f"Command: {' '.join(command)}\n")
                log_fh.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Start the process
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream and log output
                for line in iter(process.stdout.readline, ''):
                    print(line.rstrip())
                    log_fh.write(line)
                    log_fh.flush()  # Ensure log is written immediately
                
                # Wait for process to complete
                process.wait()
                
                # Log completion status
                log_fh.write(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_fh.write(f"Exit code: {process.returncode}\n")
            
            if process.returncode != 0:
                logger.error(f"Command failed with code {process.returncode}. See log: {log_file}")
            else:
                logger.info(f"Command completed successfully. Log: {log_file}")
            
            return process.returncode
        
        except FileNotFoundError as e:
            logger.error(f"Command not found: {e}")
            return 127  # Standard exit code for command not found
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            return 126  # Standard exit code for permission denied
        except KeyboardInterrupt:
            logger.warning("Command interrupted by user")
            return 130  # Standard exit code for SIGINT
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
                 serve: bool = False,
                 domain: Optional[str] = None) -> int:
        """
        Run translation (interactive or service mode).
        
        Args:
            text: Text to translate (None for interactive)
            source_lang: Source language code
            target_lang: Target language code
            serve: Start translation service
            domain: Domain for specialized translation (medical, corporate, government)
            
        Returns:
            Exit code
        """
        logger.info("🌐 Starting translation...")
        
        if serve:
            # Start translation service via the integration API
            logger.info("Starting translation service...")
            try:
                # Run the FastAPI app using uvicorn
                import uvicorn
                from integration.system import UniversalTranslationSystem as IntegrationSystem
                from integration.system_config import SystemConfig as IntegrationConfig
                system = IntegrationSystem(IntegrationConfig())
                system.initialize_all_systems()
                logger.info("Translation service started")
                return 0
            except Exception as e:
                logger.error(f"Failed to start translation service: {e}")
                return 1
        else:
            # Interactive translation via integration system
            try:
                # Initialize the integration system
                config = IntegrationConfig()
                system = IntegrationSystem(config)
                if not system.initialize_all_systems():
                    logger.error("System initialization failed")
                    return 1
                
                if text:
                    # Single translation
                    result = system.translate(text, source_lang, target_lang, domain=domain)
                    print(f"\nTranslation: {result}")
                    return 0
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
                            dom = input(f"Domain (medical/corporate/government) [{domain or 'none'}]: ") or domain
                            
                            result = system.translate(text, src, tgt, domain=dom)
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
            import numpy as np
            from integration.system import UniversalTranslationSystem as IntegrationSystem
            from integration.system_config import SystemConfig as IntegrationConfig
            
            # Initialize integration system
            config = IntegrationConfig()
            system = IntegrationSystem(config)
            if not system.initialize_all_systems():
                logger.error("Failed to initialize system for benchmarking")
                return 1
            encoder = system.encoder
            
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
                result = system.translate(text, 'en', 'es')
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
            with open(BENCHMARK_RESULTS_FILENAME, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"\nResults saved to {BENCHMARK_RESULTS_FILENAME}")
            
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
            from integration.system import UniversalTranslationSystem as IntegrationSystem
            from integration.system_config import SystemConfig as IntegrationConfig
            
            # Initialize system
            config = IntegrationConfig()
            system = IntegrationSystem(config)
            if not system.initialize_all_systems():
                logger.error("System initialization failed for export")
                return 1
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if format == 'edge':
                # Use edge model export
                system.export_edge_model(output_dir, languages=['en', 'es', 'fr', 'de'])
                return 0
            
            # For ONNX/TorchScript/TFLite, export from integration system
            output_path = Path(output_dir) / f"encoder.{format}"
            logger.info(f"Exporting to {format} not yet implemented via integration path")
            logger.info(f"Use 'python scripts/pipeline.py convert --task pytorch-to-{format}' instead")
            return 0
            
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
  
  # Translate with tone control: prefix text with [FORMAL] or [CASUAL]
  python main.py --mode translate --text "[FORMAL] How are you?" --source-lang en --target-lang es
  
  # Domain-specific translation (medical, corporate, government)
  python main.py --mode translate --text "Patient has fever" --domain medical
  
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
        default=f"{CONFIG_DIR}/{BASE_CONFIG_FILENAME}",
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
        '--domain',
        type=str,
        default=None,
        help='Domain for specialized translation (medical, corporate, government)'
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
            serve=args.serve,
            domain=args.domain
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