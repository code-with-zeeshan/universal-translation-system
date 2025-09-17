import unittest
from pathlib import Path

class TestVocabularyPath(unittest.TestCase):
    def test_vocab_dir_default_is_vocabulary(self):
        from config.schemas import VocabularyConfig
        cfg = VocabularyConfig()
        self.assertEqual(cfg.vocab_dir, "vocabulary")

    def test_main_creates_vocabulary_folder(self):
        # Run the helper that ensures data dirs exist
        from main import Path as SysPath  # reuse imported Path in main
        from main import logging
        from main import __spec__  # ensure module import side-effects are loaded
        from main import logger  # trigger logging setup already done in main
        from main import SystemIntegrator
        from main import HardwareConfig

        # Import the class that owns _check_data_availability
        from main import setup_logging  # already executed but safe to import

        # Instantiate minimal runner surrogate by mimicking the original class usage
        # The check function is module-level in main via a method on the CLI class.
        # We re-import the method owner by scanning attributes.
        import main as main_mod
        owner = None
        for name in dir(main_mod):
            obj = getattr(main_mod, name)
            if hasattr(obj, "__dict__") and callable(getattr(obj, "_check_data_availability", None)):
                owner = obj
                break
        self.assertIsNotNone(owner, "Could not locate class with _check_data_availability in main.py")

        runner = object.__new__(owner)
        # Monkey minimal attributes if needed
        # Call the method; it should create the 'vocabulary' directory
        ok = owner._check_data_availability(runner)
        self.assertTrue(ok)
        self.assertTrue(Path("vocabulary").exists(), "Expected 'vocabulary' directory to be created")

if __name__ == "__main__":
    unittest.main()