import unittest
from pathlib import Path

class TestVocabularyPath(unittest.TestCase):
    def test_vocab_dir_default_is_vocabulary(self):
        from config.schemas import VocabularyConfig
        cfg = VocabularyConfig()
        self.assertEqual(cfg.vocab_dir, "vocabulary")

if __name__ == "__main__":
    unittest.main()
