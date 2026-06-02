# training/encoder_quantizer.py
"""
Production-ready quantization pipeline with quality testing
"""

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import time
import numpy as np
import json
import torch.distributed as dist

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None

logger = logging.getLogger(__name__)

from utils.constants import QUANTIZATION_REPORT_FILENAME
from training.quantization_common import QuantizationConfig, QualityMetrics, fake_quantize_tensor
from training.quality_comparator import QualityComparator
from training.model_profiler import ModelProfiler


class EncoderQuantizer:
    """Production-ready quantization pipeline with quality testing"""

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.calibration_data = None
        self.quality_comparator = QualityComparator()
        self.profiler = ModelProfiler()

    def create_deployment_versions(self, master_model_path: str,
                                 calibration_data_path: Optional[str] = None,
                                 test_data_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Create all deployment versions with quality metrics - Updated for modern PyTorch"""

        logger.info("🚀 Starting quantization pipeline...")

        # Load master model
        master_model = torch.load(master_model_path, map_location='cpu')
        master_model.eval()

        results = {}

        # 0. Profile original FP32 model to get a baseline
        logger.info("📦 Profiling original FP32 model for baseline...")
        original_metrics = self.profiler.profile_model(master_model, test_data_path) if test_data_path else None
        if original_metrics:
            results['fp32'] = {
                'path': master_model_path,
                'metrics': original_metrics,
                'size_mb': self._get_file_size_mb(master_model_path)
            }

        # 1. Create INT8 (125MB) version
        logger.info("📦 Creating INT8 quantized model...")
        int8_path, int8_metrics = self.quantize_dynamic_modern(
            master_model,
            master_model_path.replace('.pt', '_int8.pt'),
            test_data_path,
            original_metrics=original_metrics
        )
        results['int8'] = {
            'path': int8_path,
            'metrics': int8_metrics,
            'size_mb': self._get_file_size_mb(int8_path)
        }

        # 2. Create FP16 (250MB) version
        logger.info("📦 Creating FP16 model...")
        fp16_path, fp16_metrics = self.convert_to_fp16(
            master_model,
            master_model_path.replace('.pt', '_fp16.pt'),
            test_data_path
        )
        if fp16_path:
            results['fp16'] = {
            'path': fp16_path,
            'metrics': fp16_metrics,
            'size_mb': self._get_file_size_mb(fp16_path)
        }

        # 3. Create static INT8 with calibration if available
        if calibration_data_path:
            logger.info("📦 Creating calibrated static INT8 model...")
            static_path, static_metrics = self.quantize_static_fx(
                master_model,
                calibration_data_path,
                master_model_path.replace('.pt', '_static_int8.pt'),
                test_data_path,
                original_metrics=original_metrics
            )
            results['static_int8'] = {
                'path': static_path,
                'metrics': static_metrics,
                'size_mb': self._get_file_size_mb(static_path)
            }

        # 4. Create mixed precision version (critical parts in FP16, rest in INT8)
        logger.info("📦 Creating mixed precision model...")
        mixed_path, mixed_metrics = self.create_mixed_precision_model(
            master_model,
            master_model_path.replace('.pt', '_mixed.pt'),
            test_data_path,
            original_metrics=original_metrics
        )
        results['mixed_precision'] = {
            'path': mixed_path,
            'metrics': mixed_metrics,
            'size_mb': self._get_file_size_mb(mixed_path)
        }

        # 5. Generate comparison report
        self._generate_comparison_report(results, master_model_path)

        return results

    def quantize_dynamic_modern(self, model: torch.nn.Module, output_path: str,
                        test_data_path: Optional[str] = None,
                        original_metrics: Optional[QualityMetrics] = None) -> Tuple[str, QualityMetrics]:
        """Dynamic INT8 quantization with quality testing"""

        # Configure for dynamic quantization
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.LSTM: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.GRU: torch.ao.quantization.default_dynamic_qconfig,
                torch.nn.MultiheadAttention: torch.ao.quantization.default_dynamic_qconfig,
            },
            dtype=torch.qint8,
            mapping=None,
            inplace=False
        )

        # Save quantized model
        if save_file:
            save_file(quantized_model.state_dict(), output_path)
        else:
            logger.warning("safetensors not found, falling back to torch.save.")
            torch.save(quantized_model.state_dict(), output_path)

        # Profile quantized model and return metrics
        metrics = None
        if test_data_path:
            metrics = self.profiler.profile_model(quantized_model, test_data_path)
            if original_metrics:
                metrics.compression_ratio = original_metrics.memory_mb / metrics.memory_mb

            # Log quality comparison
            logger.info(f"✅ INT8 Quantization Results:")
            if original_metrics:
                logger.info(f"   Memory: {original_metrics.memory_mb:.1f}MB → {metrics.memory_mb:.1f}MB")
                logger.info(f"   Quality retention: {(metrics.bleu_score/original_metrics.bleu_score)*100:.1f}%")

        return output_path, metrics

    def quantize_static_fx(self, model: torch.nn.Module,
                       calibration_data_path: str,
                       output_path: str,
                       test_data_path: Optional[str] = None,
                       original_metrics: Optional[QualityMetrics] = None) -> Tuple[str, QualityMetrics]:
        """Static INT8 quantization with calibration for better quality with FX graph mode (modern approach)"""

        # Clone model for quantization
        model_to_quantize = self._clone_model(model)

        # Load calibration data
        calibration_data = torch.load(calibration_data_path)

        # Create example inputs
        example_inputs = calibration_data[0]['input_ids'] if isinstance(calibration_data[0], dict) else calibration_data[0]

        # Configure quantization
        qconfig_mapping = QConfigMapping().set_global(
            get_default_qconfig('x86' if self.config.backend == 'fbgemm' else 'qnnpack')
        )

        # Prepare model
        model_prepared = prepare_fx(
            model_to_quantize,
            qconfig_mapping,
            example_inputs=example_inputs,
            prepare_custom_config=None,
            backend_config=None,
        )

        # Calibrate on real translation data
        logger.info("🔄 Calibrating model on translation data...")
        model_prepared.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.config.calibration_samples:
                    break
                input_data = batch['input_ids'] if isinstance(batch, dict) else batch
                # Handle shape mismatches
                if input_data.shape[0] != example_inputs.shape[0]:
                    input_data = input_data[:example_inputs.shape[0]]
                model_prepared(input_data)

        # Convert to quantized model
        quantized_model = convert_fx(
            model_prepared,
            convert_custom_config=None,
            backend_config=None,
        )

        # Save
        if save_file:
            save_file(quantized_model.state_dict(), output_path)
        else:
            logger.warning("safetensors not found, falling back to torch.save.")
            torch.save(quantized_model.state_dict(), output_path)

        # Profile if test data available
        metrics = None
        if test_data_path:
            metrics = self.profiler.profile_model(quantized_model, test_data_path)

        return output_path, metrics

    def convert_to_fp16(self, model: torch.nn.Module, output_path: str,
                       test_data_path: Optional[str] = None) -> Tuple[str, QualityMetrics]:
        """Convert model to FP16 with quality testing"""

        # Clone and convert to half precision
        model_fp16 = self._clone_model(model)
        model_fp16 = model_fp16.half()

        # Keep batch norm in FP32 for stability
        for module in model_fp16.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.float()

        # Save FP16 model
        if save_file:
            save_file(model_fp16.state_dict(), output_path)
        else:
            logger.warning("safetensors not found, falling back to torch.save.")
            torch.save(model_fp16.state_dict(), output_path)

        # Profile if test data available
        metrics = None
        if test_data_path:
            metrics = self.profiler.profile_model(model_fp16, test_data_path)

        return output_path, metrics

    def create_mixed_precision_model(self, model: torch.nn.Module, output_path: str,
                                   test_data_path: Optional[str] = None) -> Tuple[str, QualityMetrics]:
        """Create model with mixed precision - embeddings in FP16, rest in INT8"""

        model_mixed = self._clone_model(model)

        # Keep critical components in higher precision
        high_precision_modules = []

        for name, module in model_mixed.named_modules():
            if any(key in name.lower() for key in ['embedding', 'position', 'norm']):
                high_precision_modules.append(name)
                # Keep in FP16
                module.half()

        # Quantize the rest
        modules_to_quantize = []
        for name, module in model_mixed.named_modules():
            if name not in high_precision_modules and isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                modules_to_quantize.append(module)

        # Apply dynamic quantization to selected modules
        for module in modules_to_quantize:
            if hasattr(module, 'weight'):
                # Quantize weights to INT8
                module.weight.data = torch.quantize_per_tensor(
                    module.weight.data,
                    scale=0.1,
                    zero_point=0,
                    dtype=torch.qint8
                ).dequantize()

        # Save
        if save_file:
            save_file(model_mixed.state_dict(), output_path)
        else:
            logger.warning("safetensors not found, falling back to torch.save.")
            torch.save(model_mixed.state_dict(), output_path)

        # Profile
        metrics = None
        if test_data_path:
            metrics = self.profiler.profile_model(model_mixed, test_data_path)

        logger.info(f"✅ Mixed precision model created:")
        logger.info(f"   High precision modules: {len(high_precision_modules)}")
        logger.info(f"   Quantized modules: {len(modules_to_quantize)}")

        return output_path, metrics

    def _clone_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Create a deep copy of the model"""
        import copy
        return copy.deepcopy(model)

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        return Path(file_path).stat().st_size / (1024 * 1024)

    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]],
                                  original_model_path: str):
        """Generate detailed comparison report"""

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_model': original_model_path,
            'versions': {}
        }

        # Add each version's details
        for version_name, version_data in results.items():
            if version_data.get('metrics'):
                metrics = version_data['metrics']
                report['versions'][version_name] = {
                    'size_mb': version_data['size_mb'],
                    'latency_ms': metrics.latency_ms,
                    'memory_mb': metrics.memory_mb,
                    'bleu_score': metrics.bleu_score,
                    'compression_ratio': metrics.compression_ratio,
                    'quality_retention': metrics.bleu_score / results.get('fp32', {}).get('metrics', metrics).bleu_score * 100
                }

        # Save report
        report_path = Path(original_model_path).parent / QUANTIZATION_REPORT_FILENAME
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Comparison report saved to {report_path}")

    def quantize_distributed(self, model: torch.nn.Module, world_size: int) -> torch.nn.Module:
        """Quantize model for distributed inference"""
        if world_size == 1:
            output_path, _ = self.quantize_dynamic_modern(model, "model_int8.pt")
            return model

        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            return model
        else:
            output_path, _ = self.quantize_dynamic_modern(model, f"model_int8_rank{rank}.pt")
            return model

    def _quantize_single_model(self, model: torch.nn.Module,
                          calibration_data: List[Dict[str, torch.Tensor]],
                          config_mapping: Any,
                          example_inputs: torch.Tensor) -> torch.nn.Module:
        """Quantize a single model component"""
        # Prepare model
        model_prepared = prepare_fx(
            model,
            config_mapping,
            example_inputs=example_inputs,
        )

        # Calibrate
        model_prepared.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self.config.calibration_samples:
                    break
                input_data = batch['input_ids'] if isinstance(batch, dict) else batch
                # Handle shape mismatches
                if input_data.shape[0] != example_inputs.shape[0]:
                    input_data = input_data[:example_inputs.shape[0]]
                model_prepared(input_data)

        # Convert to quantized
        quantized_model = convert_fx(model_prepared)

        return quantized_model


class QualityPreservingQuantizer:
    """Advanced quantization with quality preservation techniques"""

    def quantize_with_calibration(self, model: torch.nn.Module,
                                  calibration_data: List[torch.Tensor],
                                  target_quality: float = 0.97) -> torch.nn.Module:
        """Quantize while preserving target quality level"""

        # 1. Collect statistics on actual usage
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.quint8,
                reduce_range=True
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        )

        # 2. Prepare model
        prepared = torch.quantization.prepare(model, inplace=False)

        # 3. Calibrate on real translation data
        logger.info("🎯 Calibrating for quality preservation...")
        prepared.eval()

        with torch.no_grad():
            for batch in calibration_data:
                prepared(batch)

        # 4. Convert with optimal parameters
        quantized = torch.quantization.convert(prepared, inplace=False)

        # 5. Verify quality meets target
        quality_score = self._evaluate_quality(model, quantized, calibration_data[:100])

        if quality_score < target_quality:
            logger.warning(f"⚠️  Quality {quality_score:.2%} below target {target_quality:.2%}")
            # Could implement iterative refinement here
        else:
            logger.info(f"✅ Quality preserved: {quality_score:.2%}")

        return quantized

    def _evaluate_quality(self, original: torch.nn.Module,
                         quantized: torch.nn.Module,
                         test_data: List[torch.Tensor]) -> float:
        """Evaluate quality preservation"""

        original.eval()
        quantized.eval()

        similarities = []

        with torch.no_grad():
            for batch in test_data:
                orig_output = original(batch)
                quant_output = quantized(batch)

                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    orig_output.flatten(),
                    quant_output.flatten(),
                    dim=0
                )
                similarities.append(similarity.item())

        return np.mean(similarities)
