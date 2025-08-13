# run_training.py
import torch
import subprocess
import sys
import os
import logging
from pathlib import Path

from utils.logging_config import setup_logging

def get_gpu_count():
    """Returns the number of available CUDA GPUs."""
    try:
        count = torch.cuda.device_count()
        if count == 0:
            logging.warning("No CUDA-enabled GPUs detected by PyTorch.")
        return count
    except Exception as e:
        logging.error(f"Error detecting GPUs: {e}")
        return 0

def get_user_choice(num_gpus):
    """Prompts the user to select the number of GPUs to use."""
    logging.info(f"\nFound {num_gpus} GPUs.")
    while True:
        choice = input(f"How many GPUs do you want to use for training? (1-{num_gpus}, or 'all') [all]: ")
        choice = choice.strip().lower() or 'all'

        if choice == 'all':
            return num_gpus
        try:
            num_to_use = int(choice)
            if 1 <= num_to_use <= num_gpus:
                return num_to_use
            else:
                logging.warning(f"❌ Invalid number. Please enter a number between 1 and {num_gpus}.")
        except ValueError:
            logging.warning("❌ Invalid input. Please enter a number or 'all'.")

def run_command(command):
    """Runs a command and streams its output."""
    logging.info(f"\n🚀 Executing command:\n{' '.join(command)}\n")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ''):
        logging.info(line.strip())
    process.wait()
    if process.returncode != 0:
        logging.error(f"\n❌ Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    """Main function to orchestrate training launch."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("--- Universal Translation System: Training Launcher ---")

    # Ensure we are at the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    num_gpus = get_gpu_count()

    if num_gpus == 0:
        logger.warning("⚠️ No GPUs detected. Running on CPU.")
        logger.info("This will be very slow. It is recommended for debugging only.")
        config_file = "config/training_cpu.yaml"
        command = [sys.executable, "training/train_universal_system.py", "--config", config_file]
        run_command(command)
    elif num_gpus == 1:
        logger.info("✅ 1 GPU detected. Starting standard training.")
        config_file = "config/training_v100.yaml" # Default to V100 for single GPU, can be made configurable
        command = [sys.executable, "training/train_universal_system.py", "--config", config_file]
        run_command(command)
    else:
        gpus_to_use = get_user_choice(num_gpus)
        if gpus_to_use == 1:
            logger.info("✅ Starting standard training on 1 GPU.")
            config_file = "config/training_v100.yaml" # Default to V100 for single GPU, can be made configurable
            command = [sys.executable, "training/train_universal_system.py", "--config", config_file]
            run_command(command)
        else:
            logger.info(f"✅ Starting distributed training on {gpus_to_use} GPUs.")
            # For multi-GPU, we assume distributed_train.py handles config selection or uses a default
            # A more advanced solution would involve dynamically selecting a config based on GPU type/count
            command = [sys.executable, "-m", "torch.distributed.launch", f"--nproc_per_node={gpus_to_use}", "training/distributed_train.py"]
            run_command(command)

    logger.info("\n🎉 Training process finished!")
