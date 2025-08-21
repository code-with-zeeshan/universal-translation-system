#!/usr/bin/env python
# scripts/config_wizard.py
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value"""
    if default:
        result = input(f"{prompt} [{default}]: ")
        return result if result else default
    else:
        return input(f"{prompt}: ")

def get_boolean_input(prompt: str, default: bool = False) -> bool:
    """Get boolean input from user"""
    default_str = "Y/n" if default else "y/N"
    result = input(f"{prompt} [{default_str}]: ").lower()
    
    if not result:
        return default
    
    return result.startswith('y')

def get_numeric_input(prompt: str, default: Optional[Union[int, float]] = None, min_value: Optional[Union[int, float]] = None, max_value: Optional[Union[int, float]] = None, is_float: bool = False) -> Union[int, float]:
    """Get numeric input from user with validation"""
    while True:
        if default is not None:
            result = input(f"{prompt} [{default}]: ")
            if not result:
                result = default
        else:
            result = input(f"{prompt}: ")
        
        try:
            if is_float:
                value = float(result)
            else:
                value = int(result)
            
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
            
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}")
                continue
            
            return value
        except ValueError:
            print("Please enter a valid number")

def detect_hardware() -> Dict[str, Any]:
    """Detect hardware capabilities"""
    result = {
        "cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": []
    }
    
    try:
        import torch
        result["gpu_available"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if result["gpu_available"]:
            for i in range(result["gpu_count"]):
                result["gpu_names"].append(torch.cuda.get_device_name(i))
                
                # Try to estimate GPU memory
                try:
                    # This is a rough estimate and may not be accurate for all GPUs
                    if "a100" in result["gpu_names"][-1].lower():
                        result["gpu_memory"].append(40)  # A100 40GB
                    elif "v100" in result["gpu_names"][-1].lower():
                        result["gpu_memory"].append(16)  # V100 16GB
                    elif "3090" in result["gpu_names"][-1].lower():
                        result["gpu_memory"].append(24)  # RTX 3090 24GB
                    elif "2080" in result["gpu_names"][-1].lower():
                        result["gpu_memory"].append(8)   # RTX 2080 8GB
                    else:
                        result["gpu_memory"].append(8)   # Default assumption
                except:
                    result["gpu_memory"].append(8)  # Default assumption
    except ImportError:
        pass
    
    return result

def create_training_config(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Create training configuration based on hardware and user input"""
    print("\n=== Training Configuration ===")
    
    # Determine batch size based on GPU
    default_batch_size = 16
    if hardware["gpu_available"]:
        gpu_name = hardware["gpu_names"][0].lower() if hardware["gpu_names"] else ""
        if "a100" in gpu_name:
            default_batch_size = 64
        elif "v100" in gpu_name:
            default_batch_size = 32
        elif "3090" in gpu_name:
            default_batch_size = 24
    
    batch_size = get_numeric_input("Batch size", default_batch_size, min_value=1)
    
    # Get other training parameters
    epochs = get_numeric_input("Number of epochs", 20, min_value=1)
    learning_rate = get_numeric_input("Learning rate", 5e-5, min_value=1e-7, max_value=1.0, is_float=True)
    
    # Determine optimizer
    optimizer_options = ["adam", "adamw", "sgd"]
    print("Available optimizers:")
    for i, opt in enumerate(optimizer_options):
        print(f"  {i+1}. {opt}")
    
    optimizer_idx = get_numeric_input("Select optimizer (number)", 2, min_value=1, max_value=len(optimizer_options)) - 1
    optimizer = optimizer_options[optimizer_idx]
    
    # Get language pairs
    default_langs = "en-es,en-fr,en-de,en-zh"
    langs = get_user_input("Language pairs (comma-separated, e.g., en-es,en-fr)", default_langs)
    lang_pairs = [pair.strip() for pair in langs.split(",")]
    
    # Advanced options
    print("\nAdvanced training options:")
    use_mixed_precision = get_boolean_input("Use mixed precision training", True)
    use_gradient_accumulation = get_boolean_input("Use gradient accumulation", False)
    
    gradient_accumulation_steps = 1
    if use_gradient_accumulation:
        gradient_accumulation_steps = get_numeric_input("Gradient accumulation steps", 4, min_value=1)
    
    use_lr_scheduler = get_boolean_input("Use learning rate scheduler", True)
    lr_scheduler_type = "linear"
    if use_lr_scheduler:
        lr_scheduler_options = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        print("Available learning rate schedulers:")
        for i, scheduler in enumerate(lr_scheduler_options):
            print(f"  {i+1}. {scheduler}")
        
        scheduler_idx = get_numeric_input("Select scheduler (number)", 1, min_value=1, max_value=len(lr_scheduler_options)) - 1
        lr_scheduler_type = lr_scheduler_options[scheduler_idx]
    
    # Create configuration
    training_config = {
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "language_pairs": lang_pairs,
            "use_gpu": hardware["gpu_available"],
            "distributed": hardware["gpu_count"] > 1,
            "mixed_precision": use_mixed_precision,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
    }
    
    if use_lr_scheduler:
        training_config["training"]["lr_scheduler"] = {
            "type": lr_scheduler_type,
            "warmup_steps": 1000 if "warmup" in lr_scheduler_type else 0
        }
    
    return training_config

def create_model_config() -> Dict[str, Any]:
    """Create model configuration based on user input"""
    print("\n=== Model Configuration ===")
    
    # Basic model parameters
    encoder_dim = get_numeric_input("Encoder dimension", 1024, min_value=64)
    decoder_dim = get_numeric_input("Decoder dimension", 512, min_value=64)
    num_layers = get_numeric_input("Number of layers", 6, min_value=1)
    num_heads = get_numeric_input("Number of attention heads", 8, min_value=1)
    
    # Advanced options
    print("\nAdvanced model options:")
    use_quantization = get_boolean_input("Use quantization", False)
    use_pruning = get_boolean_input("Use pruning", False)
    
    quantization_config = None
    if use_quantization:
        quantization_options = ["dynamic", "static", "qat"]
        print("Available quantization methods:")
        for i, method in enumerate(quantization_options):
            print(f"  {i+1}. {method}")
        
        method_idx = get_numeric_input("Select quantization method (number)", 1, min_value=1, max_value=len(quantization_options)) - 1
        quantization_method = quantization_options[method_idx]
        
        quantization_config = {
            "method": quantization_method,
            "bits": 8
        }
    
    pruning_config = None
    if use_pruning:
        pruning_options = ["magnitude", "l1_unstructured", "l2_structured"]
        print("Available pruning methods:")
        for i, method in enumerate(pruning_options):
            print(f"  {i+1}. {method}")
        
        method_idx = get_numeric_input("Select pruning method (number)", 1, min_value=1, max_value=len(pruning_options)) - 1
        pruning_method = pruning_options[method_idx]
        
        pruning_config = {
            "method": pruning_method,
            "sparsity": 0.5
        }
    
    # Create configuration
    model_config = {
        "model": {
            "encoder_dim": encoder_dim,
            "decoder_dim": decoder_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_length": 256,
            "vocab_size": 50000
        }
    }
    
    if use_quantization:
        model_config["model"]["quantization"] = quantization_config
    
    if use_pruning:
        model_config["model"]["pruning"] = pruning_config
    
    return model_config

def create_deployment_config() -> Dict[str, Any]:
    """Create deployment configuration based on user input"""
    print("\n=== Deployment Configuration ===")
    
    deployment_options = ["docker", "kubernetes", "local"]
    print("Available deployment types:")
    for i, deploy_type in enumerate(deployment_options):
        print(f"  {i+1}. {deploy_type}")
    
    type_idx = get_numeric_input("Select deployment type (number)", 1, min_value=1, max_value=len(deployment_options)) - 1
    deployment_type = deployment_options[type_idx]
    
    if deployment_type == "kubernetes":
        namespace = get_user_input("Kubernetes namespace", "universal-translation")
        replicas = get_numeric_input("Number of decoder replicas", 3, min_value=1)
        
        # Resource requests and limits
        print("\nResource requests and limits:")
        cpu_request = get_user_input("CPU request", "2")
        memory_request = get_user_input("Memory request", "4Gi")
        gpu_request = get_numeric_input("GPU request", 1, min_value=0)
        
        return {
            "deployment": {
                "type": deployment_type,
                "kubernetes": {
                    "namespace": namespace,
                    "decoder_replicas": replicas,
                    "use_gpu": gpu_request > 0,
                    "resource_requests": {
                        "cpu": cpu_request,
                        "memory": memory_request,
                        "gpu": gpu_request
                    },
                    "resource_limits": {
                        "cpu": cpu_request,
                        "memory": memory_request,
                        "gpu": gpu_request
                    }
                }
            }
        }
    elif deployment_type == "docker":
        port = get_numeric_input("API port", 8000, min_value=1, max_value=65535)
        use_gpu = get_boolean_input("Use GPU (if available)", True)
        
        return {
            "deployment": {
                "type": deployment_type,
                "docker": {
                    "api_port": port,
                    "use_gpu": use_gpu,
                    "compose_file": "docker-compose.yml"
                }
            }
        }
    else:
        port = get_numeric_input("API port", 8000, min_value=1, max_value=65535)
        use_gpu = get_boolean_input("Use GPU (if available)", True)
        
        return {
            "deployment": {
                "type": deployment_type,
                "local": {
                    "api_port": port,
                    "use_gpu": use_gpu,
                    "log_dir": "logs"
                }
            }
        }

def create_monitoring_config() -> Dict[str, Any]:
    """Create monitoring configuration based on user input"""
    print("\n=== Monitoring Configuration ===")
    
    use_monitoring = get_boolean_input("Enable monitoring", True)
    
    if not use_monitoring:
        return {}
    
    use_prometheus = get_boolean_input("Use Prometheus", True)
    use_grafana = get_boolean_input("Use Grafana", True)
    
    prometheus_port = 9090
    grafana_port = 3000
    
    if use_prometheus:
        prometheus_port = get_numeric_input("Prometheus port", 9090, min_value=1, max_value=65535)
    
    if use_grafana:
        grafana_port = get_numeric_input("Grafana port", 3000, min_value=1, max_value=65535)
    
    return {
        "monitoring": {
            "enabled": True,
            "prometheus": {
                "enabled": use_prometheus,
                "port": prometheus_port
            },
            "grafana": {
                "enabled": use_grafana,
                "port": grafana_port
            },
            "metrics": {
                "collect_system_metrics": True,
                "collect_gpu_metrics": True,
                "collect_translation_metrics": True
            }
        }
    }

def create_logging_config() -> Dict[str, Any]:
    """Create logging configuration based on user input"""
    print("\n=== Logging Configuration ===")
    
    log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    print("Available log levels:")
    for i, level in enumerate(log_level_options):
        print(f"  {i+1}. {level}")
    
    level_idx = get_numeric_input("Select log level (number)", 2, min_value=1, max_value=len(log_level_options)) - 1
    log_level = log_level_options[level_idx]
    
    log_dir = get_user_input("Log directory", "logs")
    
    return {
        "logging": {
            "level": log_level,
            "dir": log_dir,
            "file_rotation": True,
            "max_file_size_mb": 10,
            "max_files": 5
        }
    }

def main():
    print("=== Universal Translation System Configuration Wizard ===")
    print("This wizard will help you create a valid configuration file.")
    
    # Detect hardware
    print("\nDetecting hardware capabilities...")
    hardware = detect_hardware()
    print(f"CPU cores: {hardware['cpu_count']}")
    if hardware["gpu_available"]:
        print(f"GPUs: {hardware['gpu_count']} ({', '.join(hardware['gpu_names'])})")
        for i, (name, memory) in enumerate(zip(hardware['gpu_names'], hardware['gpu_memory'])):
            print(f"  GPU {i}: {name} (estimated {memory}GB memory)")
    else:
        print("No GPUs detected")
    
    # Get configuration type
    config_type_options = ["training", "inference", "all"]
    print("\nAvailable configuration types:")
    for i, config_type in enumerate(config_type_options):
        print(f"  {i+1}. {config_type}")
    
    type_idx = get_numeric_input("Select configuration type (number)", 3, min_value=1, max_value=len(config_type_options)) - 1
    config_type = config_type_options[type_idx]
    
    # Create configuration
    config = {}
    
    if config_type in ["training", "all"]:
        config.update(create_training_config(hardware))
    
    if config_type in ["all"]:
        config.update(create_model_config())
    
    if config_type in ["inference", "all"]:
        config.update(create_deployment_config())
    
    # Add monitoring and logging
    config.update(create_monitoring_config())
    config.update(create_logging_config())
    
    # Add version info
    config["version"] = "1.0.0"
    
    # Write configuration to file
    output_path = get_user_input("Output file path", "config/generated_config.yaml")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write as YAML or JSON based on file extension
    if output_path.endswith('.json'):
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Configuration written to {output_path}")
    print("You can validate this configuration with: python scripts/validate_config.py", output_path)
    
    # Offer to validate the configuration
    if get_boolean_input("Validate the configuration now?", True):
        try:
            # Import the validation script
            sys.path.insert(0, str(project_root / "scripts"))
            import validate_config
            
            # Run validation
            schema_errors = validate_config.validate_config(output_path)
            if schema_errors:
                print("❌ Schema validation errors:")
                for error in schema_errors:
                    print(f"  - {error}")
            else:
                print("✅ Schema validation passed")
            
            # Check references
            reference_errors = validate_config.check_config_references(output_path)
            if reference_errors:
                print("⚠️ Reference validation warnings:")
                for error in reference_errors:
                    print(f"  - {error}")
            else:
                print("✅ Reference validation passed")
            
            # Check consistency
            consistency_errors = validate_config.check_config_consistency(output_path)
            if consistency_errors:
                print("⚠️ Consistency warnings:")
                for error in consistency_errors:
                    print(f"  - {error}")
            else:
                print("✅ Consistency check passed")
            
            # Suggest improvements
            suggestions = validate_config.suggest_improvements(output_path)
            if suggestions:
                print("\n💡 Suggestions for improvement:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
            
            if not schema_errors and not reference_errors and not consistency_errors:
                print("\n✅ Configuration is valid and ready to use!")
            
        except ImportError:
            print("❌ Could not import validation script. Please run validation manually.")
        except Exception as e:
            print(f"❌ Error during validation: {e}")

if __name__ == "__main__":
    main()