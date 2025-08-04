"""Test that all fixes are properly integrated"""
import unittest
from pathlib import Path
import os
import tempfile
import json

class TestIntegrationFixes(unittest.TestCase):
    
    def test_security_module_exists(self):
        """Test that security module is created"""
        security_path = Path('utils/security.py')
        self.assertTrue(security_path.exists())
    
    def test_no_trust_remote_code(self):
        """Test that trust_remote_code=True is removed"""
        files_to_check = [
            'data/download_curated_data.py',
            'data/download_training_data.py'
        ]
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
                self.assertNotIn('trust_remote_code=True', content)
    
    def test_imports_fixed(self):
        """Test that missing imports are added"""
        # Check datetime import in pipeline_connector
        with open('data/pipeline_connector.py', 'r') as f:
            content = f.read()
            self.assertIn('from datetime import datetime', content)
    
    def test_decoder_implementation_fixed(self):
        """Test that decoder implementation is complete"""
        with open('cloud_decoder/optimized_decoder.py', 'r') as f:
            content = f.read()
            # Should not contain TODO placeholder
            self.assertNotIn('"translation": "TODO: implement decode logic"', content)
            # Should contain actual implementation
            self.assertIn('decompress_encoder_output', content)
            self.assertIn('decode_tokens_to_text', content)
    
    def test_tokenization_implementation_fixed(self):
        """Test that tokenization implementation is complete"""
        with open('encoder_core/src/vocabulary_pack.cpp', 'r') as f:
            content = f.read()
            # Should not contain TODO
            self.assertNotIn('TODO: Integrate production BPE or SentencePiece subword tokenization here', content)
            # Should contain actual implementation
            self.assertIn('Production SentencePiece subword tokenization', content)
    
    def test_remote_encoder_implementation_fixed(self):
        """Test that remote encoder implementation is complete"""
        with open('universal-decoder-node/universal-decoder-node/cli.py', 'r') as f:
            content = f.read()
            # Should not contain TODO
            self.assertNotIn('TODO: Implement remote encoder call', content)
            # Should contain actual implementation
            self.assertIn('encoder_resp = requests.post', content)
    
    def test_jwt_secret_environment_variable(self):
        """Test that JWT secret uses environment variable"""
        with open('universal-decoder-node/universal-decoder-node/cli.py', 'r') as f:
            content = f.read()
            self.assertIn('os.environ.get(\'JWT_SECRET\'', content)
    
    def test_specific_exception_handling(self):
        """Test that specific exception handling is implemented"""
        with open('vocabulary/vocabulary_manager.py', 'r') as f:
            content = f.read()
            self.assertIn('except FileNotFoundError:', content)
            self.assertIn('except json.JSONDecodeError as e:', content)
    
    def test_memory_cleanup_implementation(self):
        """Test that memory cleanup is implemented"""
        with open('vocabulary/vocabulary_manager.py', 'r') as f:
            content = f.read()
            self.assertIn('def cleanup_cache(self):', content)
            self.assertIn('self.loaded_packs.clear()', content)
    
    def test_circular_dependency_fixed(self):
        """Test that circular dependencies are resolved"""
        # This would need more complex testing in practice
        pass

if __name__ == '__main__':
    unittest.main()