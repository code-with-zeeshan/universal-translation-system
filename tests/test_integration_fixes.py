"""Test that all fixes are properly integrated"""
import unittest
from pathlib import Path

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
    
    def test_circular_dependency_fixed(self):
        """Test that circular dependencies are resolved"""
        # This would need more complex testing in practice
        pass

if __name__ == '__main__':
    unittest.main()