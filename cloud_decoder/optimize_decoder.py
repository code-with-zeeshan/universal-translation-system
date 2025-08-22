#!/usr/bin/env python
"""
Optimize the decoder model for better performance.
Supports quantization, batching, and mixed precision.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from config.config_models import DecoderConfig, load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DecoderOptimizer:
    """Optimize decoder model for better performance."""
    
    def __init__(self, config: DecoderConfig):
        """
        Initialize the optimizer.
        
        Args:
            config: Decoder configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        
        logger.info(f"Initializing decoder optimizer with device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Load the decoder model.
        
        Args:
            model_path: Path to model file (overrides config)
            
        Returns:
            Loaded model
        """
        path = model_path or self.config.model_path
        logger.info(f"Loading model from {path}")
        
        try:
            self.model = torch.load(path, map_location=self.device)
            self.model.eval()
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def optimize_for_inference(self) -> nn.Module:
        """
        Optimize model for inference.
        
        Returns:
            Optimized model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Optimizing model for inference")
        
        # Fuse batch normalization layers
        self._fuse_batch_norm()
        
        # Convert to TorchScript
        self._convert_to_torchscript()
        
        return self.model
    
    def _fuse_batch_norm(self) -> None:
        """Fuse batch normalization layers."""
        logger.info("Fusing batch normalization layers")
        
        # This is a placeholder for actual implementation
        # In a real implementation, you would iterate through the model
        # and fuse Conv2d+BatchNorm2d or Linear+BatchNorm1d layers
        pass
    
    def _convert_to_torchscript(self) -> None:
        """Convert model to TorchScript."""
        logger.info("Converting model to TorchScript")
        
        try:
            # Create example inputs
            batch_size = self.config.batch_size
            seq_len = 32  # Example sequence length
            embedding_dim = 768  # Example embedding dimension
            
            example_input = torch.randn(
                batch_size, seq_len, embedding_dim, device=self.device
            )
            
            # Trace the model
            self.model = torch.jit.trace(self.model, example_input)
            logger.info("Model successfully converted to TorchScript")
        except Exception as e:
            logger.error(f"Error converting to TorchScript: {e}")
            logger.warning("Continuing with non-TorchScript model")
    
    def quantize(self, quantization_type: str = "dynamic") -> nn.Module:
        """
        Quantize the model.
        
        Args:
            quantization_type: Type of quantization (dynamic, static, or qat)
            
        Returns:
            Quantized model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Quantizing model with {quantization_type} quantization")
        
        if quantization_type == "dynamic":
            self._dynamic_quantization()
        elif quantization_type == "static":
            self._static_quantization()
        elif quantization_type == "qat":
            logger.warning("Quantization-aware training not implemented")
        else:
            logger.warning(f"Unknown quantization type: {quantization_type}")
        
        return self.model
    
    def _dynamic_quantization(self) -> None:
        """Apply dynamic quantization."""
        logger.info("Applying dynamic quantization")
        
        try:
            # Check if model is TorchScript
            is_torchscript = isinstance(self.model, torch.jit.ScriptModule)
            
            if is_torchscript:
                logger.warning("Dynamic quantization not supported for TorchScript models")
                return
            
            # Apply dynamic quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization applied successfully")
        except Exception as e:
            logger.error(f"Error applying dynamic quantization: {e}")
            logger.warning("Continuing with non-quantized model")
    
    def _static_quantization(self) -> None:
        """Apply static quantization."""
        logger.info("Static quantization requires calibration data")
        logger.warning("Static quantization not implemented")
    
    def benchmark(self, 
                 batch_sizes: List[int] = None, 
                 sequence_lengths: List[int] = None,
                 num_runs: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark model performance.
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_runs: Number of runs for each configuration
            
        Returns:
            Dictionary of benchmark results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64]
        sequence_lengths = sequence_lengths or [16, 32, 64, 128, 256]
        
        logger.info(f"Benchmarking model with {num_runs} runs per configuration")
        
        results = {}
        
        for batch_size in batch_sizes:
            batch_results = {}
            
            for seq_len in sequence_lengths:
                # Create random input
                embedding_dim = 768  # Example embedding dimension
                input_tensor = torch.randn(
                    batch_size, seq_len, embedding_dim, device=self.device
                )
                
                # Warm-up
                for _ in range(3):
                    with torch.no_grad():
                        self.model(input_tensor)
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    with torch.no_grad():
                        self.model(input_tensor)
                
                # Synchronize if using GPU
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_runs
                
                batch_results[f"seq_len_{seq_len}"] = avg_time
                
                logger.info(
                    f"Batch size: {batch_size}, Sequence length: {seq_len}, "
                    f"Average time: {avg_time:.4f} seconds"
                )
            
            results[f"batch_size_{batch_size}"] = batch_results
        
        return results
    
    def save_optimized_model(self, output_path: Optional[str] = None) -> str:
        """
        Save the optimized model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Default output path
        if output_path is None:
            model_dir = Path(self.config.model_path).parent
            model_name = Path(self.config.model_path).stem
            output_path = str(model_dir / f"{model_name}_optimized.pt")
        
        logger.info(f"Saving optimized model to {output_path}")
        
        try:
            torch.save(self.model, output_path)
            logger.info("Model saved successfully")
            return output_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimize decoder model")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )
    parser.add_argument(
        "--model", type=str, help="Path to model file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save optimized model"
    )
    parser.add_argument(
        "--quantize", choices=["dynamic", "static", "qat"], 
        help="Apply quantization"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], 
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config).decoder
    else:
        config = DecoderConfig()
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Initialize optimizer
    optimizer = DecoderOptimizer(config)
    
    # Load model
    optimizer.load_model(args.model)
    
    # Optimize for inference
    optimizer.optimize_for_inference()
    
    # Apply quantization if requested
    if args.quantize:
        optimizer.quantize(args.quantize)
    
    # Run benchmark if requested
    if args.benchmark:
        optimizer.benchmark()
    
    # Save optimized model
    optimizer.save_optimized_model(args.output)


if __name__ == "__main__":
    main()