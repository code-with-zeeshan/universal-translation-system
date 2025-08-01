"""
Complete integration test suite
"""
import unittest
import tempfile
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

class TestCompleteIntegration(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_security_module_integration(self):
        """Test security module is properly integrated"""
        from utils.security import validate_model_source, validate_path_component
        
        # Test trusted sources
        self.assertTrue(validate_model_source('facebook/nllb-200'))
        self.assertTrue(validate_model_source('Helsinki-NLP/opus-mt'))
        
        # Test untrusted sources
        self.assertFalse(validate_model_source('random/unknown-model'))
        
        # Test path validation
        self.assertEqual(validate_path_component('valid_name'), 'valid_name')
        
        with self.assertRaises(ValueError):
            validate_path_component('../etc/passwd')
    
    def test_no_trust_remote_code(self):
        """Verify trust_remote_code=True is removed"""
        files_to_check = [
            'data/download_curated_data.py',
            'data/download_training_data.py',
            'training/bootstrap_from_pretrained.py'
        ]
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    self.assertNotIn('trust_remote_code=True', content, 
                                   f"Found trust_remote_code=True in {file_path}")
    
    def test_all_imports_resolved(self):
        """Test all imports are properly resolved"""
        try:
            # Test critical imports
            from utils.security import validate_model_source
            from utils.base_classes import BaseDataProcessor
            from utils.dataset_classes import ModernParallelDataset
            from utils.config_validator import ConfigValidator
            
            # Test module imports
            from data.pipeline_connector import PipelineConnector
            from vocabulary.vocabulary_manager import VocabularyManager
            from integration.connect_all_systems import UniversalTranslationSystem
            
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_circular_dependencies_resolved(self):
        """Test no circular dependencies exist"""
        # This would need more sophisticated testing in practice
        try:
            # Import in different orders
            from data.vocabulary_connector import VocabularyConnector
            from vocabulary.create_vocabulary_packs_from_data import VocabularyPackCreator
            
            # Try reverse order
            import importlib
            importlib.reload(sys.modules['vocabulary.create_vocabulary_packs_from_data'])
            importlib.reload(sys.modules['data.vocabulary_connector'])
            
            self.assertTrue(True, "No circular dependency detected")
        except ImportError as e:
            self.fail(f"Circular dependency detected: {e}")
    
    def test_dataset_classes_available(self):
        """Test dataset classes are properly defined"""
        from utils.dataset_classes import ModernParallelDataset, StreamingParallelDataset
        
        # Test instantiation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello\tHola\ten\tes\n")
            f.write("World\tMundo\ten\tes\n")
            temp_file = f.name
        
        try:
            dataset = ModernParallelDataset(temp_file)
            self.assertEqual(len(dataset), 2)
            
            # Test streaming dataset
            streaming_dataset = StreamingParallelDataset(temp_file)
            items = list(streaming_dataset)
            self.assertEqual(len(items), 2)
        finally:
            Path(temp_file).unlink()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_modules_integration(self):
        """Test training modules are properly integrated"""
        try:
            # Test imports
            from training.distributed_train import UnifiedDistributedTrainer, TrainingConfig
            from training.memory_efficient_training import MemoryOptimizedTrainer, MemoryConfig
            from training.quantization_pipeline import EncoderQuantizer, QuantizationConfig
            from training.progressive_training import ProgressiveTrainingStrategy
            from training.training_validator import TrainingValidator
            from training.training_utils import get_optimal_batch_size
        
            # Test configurations
            training_config = TrainingConfig()
            memory_config = MemoryConfig()
            quant_config = QuantizationConfig()
        
            # Validate configs
            self.assertIsNotNone(training_config)
            self.assertIsNotNone(memory_config)
            self.assertIsNotNone(quant_config)
        
        except ImportError as e:
            self.fail(f"Training module import failed: {e}")

    def test_dataset_functionality(self):
        """Test dataset classes work correctly"""
        from utils.dataset_classes import ModernParallelDataset
    
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world\tHola mundo\ten\tes\n")
            f.write("Good morning\tBuenos días\ten\tes\n")
            temp_file = f.name
    
        try:
            # Create dataset
            dataset = ModernParallelDataset(temp_file)
        
            # Test length
            self.assertEqual(len(dataset), 2)
        
            # Test item access
            item = dataset[0]
            self.assertIn('source_ids', item)
            self.assertIn('target_ids', item)
            self.assertIn('metadata', item)
        
            # Test metadata
            self.assertEqual(item['metadata']['source_lang'], 'en')
            self.assertEqual(item['metadata']['target_lang'], 'es')
        
        finally:
            Path(temp_file).unlink()

    def test_no_undefined_references(self):
        """Test that all referenced methods/classes are defined"""
        # This would need static analysis tools in practice
        # For now, just test critical imports work
    
        from training.quantization_pipeline import EncoderQuantizer
        quantizer = EncoderQuantizer()
    
        # Check methods exist
        self.assertTrue(hasattr(quantizer, 'quantize_dynamic_modern'))
        self.assertTrue(hasattr(quantizer, '_quantize_single_model'))
        self.assertTrue(hasattr(quantizer, 'create_deployment_versions'))    

if __name__ == '__main__':
    unittest.main()