# training/convert_models.py 
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """Updated model conversion with modern practices"""
    
    @staticmethod
    def pytorch_to_onnx(model_path: str, 
                       output_path: str,
                       dummy_input: torch.Tensor,
                       opset_version: int = 17,
                       use_dynamo: bool = True) -> bool:
        """Convert PyTorch model to ONNX with modern exporter"""
        
        try:
            # Load model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Check PyTorch version and use appropriate export
            pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        
            if pytorch_version >= (2, 1) and use_dynamo:
                # Use new torch.onnx.dynamo_export for PyTorch 2.1+
                try:
                    logger.info("Using torch.onnx.dynamo_export (PyTorch 2.1+)...")
                
                    # Create export program
                    export_program = torch.export.export(
                        model,
                        (dummy_input,)
                    )
                
                    # Convert to ONNX
                    import torch.onnx._dynamo as dynamo_onnx
                
                    dynamo_onnx.export(
                       export_program,
                       dummy_input,
                       output_path,
                       opset_version=opset_version,
                       input_names=['input_ids'],
                       output_names=['encoder_output']
                    )
                
                except Exception as e:
                    logger.warning(f"New export failed: {e}, falling back to legacy export")
                    use_dynamo = False
        
            if not use_dynamo or pytorch_version < (2, 1):
                # Use legacy export without deprecated parameters
                logger.info("Using legacy ONNX export...")
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input_ids'],
                    output_names=['encoder_output'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'sequence'},
                        'encoder_output': {0: 'batch', 1: 'sequence'}
                    },
                    # Removed deprecated parameters:
                    # operator_export_type - removed
                    # enable_onnx_checker - removed
                    # keep_initializers_as_inputs - removed
                    export_params=True,
                    verbose=False
                )
        
            logger.info(f"✅ ONNX model exported to {output_path}")
            
            # Optimize with onnx-simplifier if available
            try:
                import onnx
                from onnxsim import simplify
                
                logger.info("Optimizing ONNX model...")
                onnx_model = onnx.load(output_path)
                model_sim, check = simplify(onnx_model)
                if check:
                    onnx.save(model_sim, output_path)
                    logger.info("ONNX model optimized successfully")
                else:
                    logger.warning("⚠️ ONNX optimization failed validation")
                    
            except ImportError:
                logger.warning("onnx-simplifier not available, skipping optimization")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def onnx_to_coreml(onnx_path: str, 
                      output_path: str,
                      minimum_deployment_target: str = "15.0") -> bool:
        """Convert ONNX to CoreML with modern practices"""
        
        try:
            import coremltools as ct
            
            logger.info("Converting ONNX to CoreML...")
            
            # Modern CoreML conversion
            model = ct.convert(
                onnx_path,
                minimum_deployment_target=minimum_deployment_target,
                compute_precision=ct.precision.FLOAT16,
                convert_to="mlprogram",  # Modern format (not "neuralnetwork")
                compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
                source="pytorch"  # Specify source framework
            )
            
            # Add metadata
            model.short_description = "Universal Translation Model"
            model.version = "1.0.0"
            
            model.save(output_path)
            logger.info(f"CoreML model saved to {output_path}")
            return True
            
        except ImportError:
            logger.error("coremltools not available")
            return False
        except Exception as e:
            logger.error(f"CoreML conversion failed: {e}")
            return False
    
    @staticmethod
    def onnx_to_tflite(onnx_path: str, 
                      output_path: str,
                      quantize: bool = True) -> bool:
        """Convert ONNX to TFLite with modern practices"""
        
        try:
            import tensorflow as tf
            
            # Modern approach: Use tf2onnx for better compatibility
            logger.info("Converting ONNX to TensorFlow...")
            
            # Use tf2onnx for better conversion
            try:
                import tf2onnx
                import onnx
                
                onnx_model = onnx.load(onnx_path)
                
                # Convert ONNX to TF SavedModel
                tf_model, _ = tf2onnx.convert.from_onnx(
                    onnx_model,
                    input_names=None,
                    output_names=None,
                    opset=17
                )
                
                # Save as SavedModel first
                temp_saved_model = "temp_saved_model"
                tf.saved_model.save(tf_model, temp_saved_model)
                
                # Convert to TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model)
                
                # Modern optimization settings
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                if quantize:
                    # Modern quantization
                    converter.target_spec.supported_types = [tf.float16]
                    converter.representative_dataset = ModelConverter._representative_dataset
                
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                
                # Enable experimental features
                converter.experimental_new_converter = True
                converter.experimental_new_quantizer = True
                
                tflite_model = converter.convert()
                
                # Save TFLite model
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                logger.info(f"TFLite model saved to {output_path}")
                
                # Cleanup
                import shutil
                shutil.rmtree(temp_saved_model, ignore_errors=True)
                
                return True
                
            except ImportError:
                logger.error("tf2onnx not available, cannot convert ONNX to TFLite")
                return False
                
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            return False
    
    @staticmethod
    def _representative_dataset():
        """Representative dataset for quantization"""
        # Generate dummy data for quantization
        for _ in range(100):
            yield [torch.randn(1, 512).numpy().astype('float32')]
    
    @staticmethod
    def pytorch_to_tensorrt(model_path: str, 
                           output_path: str,
                           input_shape: tuple,
                           precision: str = "fp16") -> bool:
        """Convert PyTorch to TensorRT (modern alternative)"""
        
        try:
            import torch_tensorrt
            
            logger.info("Converting PyTorch to TensorRT...")
            
            model = torch.load(model_path)
            model.eval()
            
            # Trace the model
            example_input = torch.randn(*input_shape).cuda()
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert to TensorRT
            if precision == "fp16":
                enabled_precisions = {torch.float, torch.half}
            elif precision == "int8":
                enabled_precisions = {torch.float, torch.half, torch.int8}
            else:
                enabled_precisions = {torch.float}
            
            trt_model = torch_tensorrt.compile(
                traced_model,
                inputs=[torch_tensorrt.Input(input_shape)],
                enabled_precisions=enabled_precisions,
                workspace_size=1 << 30,  # 1GB
                max_batch_size=32
            )
            
            # Save TensorRT model
            torch.jit.save(trt_model, output_path)
            logger.info(f"TensorRT model saved to {output_path}")
            
            return True
            
        except ImportError:
            logger.error("torch_tensorrt not available")
            return False
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    @staticmethod
    def validate_conversion(original_model_path: str, 
                          converted_model_path: str,
                          model_type: str = "onnx") -> bool:
        """Validate converted model accuracy"""
        
        try:
            logger.info(f"Validating {model_type} conversion...")
            
            # Load original model
            original_model = torch.load(original_model_path)
            original_model.eval()
            
            # Create test input
            test_input = torch.randn(1, 512)
            
            # Get original output
            with torch.no_grad():
                original_output = original_model(test_input)
            
            # Test converted model based on type
            if model_type == "onnx":
                import onnxruntime as ort
                
                session = ort.InferenceSession(converted_model_path)
                converted_output = session.run(
                    None, 
                    {"input_ids": test_input.numpy()}
                )[0]
                
                # Compare outputs
                diff = torch.abs(original_output - torch.tensor(converted_output))
                max_diff = torch.max(diff).item()
                
                logger.info(f"Maximum difference: {max_diff}")
                return max_diff < 1e-4
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

# Usage example with modern practices
if __name__ == "__main__":
    converter = ModelConverter()
    
    # Modern conversion pipeline
    model_path = "models/universal_encoder.pt"
    dummy_input = torch.randn(1, 512)
    
    # Convert to ONNX with modern exporter
    success = converter.pytorch_to_onnx(
        model_path, 
        "models/universal_encoder.onnx",
        dummy_input,
        use_dynamo=True
    )
    
    if success:
        # Validate conversion
        converter.validate_conversion(
            model_path,
            "models/universal_encoder.onnx",
            "onnx"
        )
        
        # Convert to mobile formats
        converter.onnx_to_coreml(
            "models/universal_encoder.onnx",
            "models/universal_encoder.mlmodel"
        )
        
        converter.onnx_to_tflite(
            "models/universal_encoder.onnx",
            "models/universal_encoder.tflite"
        )