import os
import tempfile
import msgpack
import unittest
from pathlib import Path

from config.schemas import RootConfig, DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig
from runtime.vocabulary.manager import UnifiedVocabularyManager, VocabularyMode


def _write_pack(dir_path: Path, name: str, version: str = "1.0", languages=None, tokens=None, subwords=None, special_tokens=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    pack_path = dir_path / f"{name}_v{version}.msgpack"
    payload = {
        "name": name,
        "version": version,
        "languages": languages or [],
        "tokens": tokens or {"hello": 10, "world": 11, "test": 12},
        "subwords": subwords or {"##ing": 1001, "##ed": 1002},
        "special_tokens": special_tokens or {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3},
    }
    pack_path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    return pack_path


class TestVocabularyLRUAndFallback(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.vocab_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_config(self, mapping):
        return RootConfig(
            data=DataConfig(training_distribution={}),
            model=ModelConfig(),
            training=TrainingConfig(),
            memory=MemoryConfig(),
            vocabulary=VocabularyConfig(language_to_pack_mapping=mapping, vocab_dir=str(self.vocab_dir))
        )

    def test_lru_eviction_order(self):
        # Create three packs
        _write_pack(self.vocab_dir, "pack1")
        _write_pack(self.vocab_dir, "pack2")
        _write_pack(self.vocab_dir, "pack3")

        cfg = self._make_config({"l1": "pack1", "l2": "pack2", "l3": "pack3"})

        # Small cache to force eviction
        manager = UnifiedVocabularyManager(config=cfg, vocab_dir=str(self.vocab_dir), mode=VocabularyMode.OPTIMIZED, cache_size=2)

        # Load pack1 and pack2
        p1 = manager.get_vocab_for_pair("l1", "l1")
        p2 = manager.get_vocab_for_pair("l2", "l2")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertIn("pack1", manager._vocabulary_cache)
        self.assertIn("pack2", manager._vocabulary_cache)

        # Touch pack1 to make it most recently used
        _ = manager.get_vocab_for_pair("l1", "l1")

        # Load pack3 -> should evict least recently used (pack2)
        p3 = manager.get_vocab_for_pair("l3", "l3")
        self.assertIsNotNone(p3)

        self.assertIn("pack1", manager._vocabulary_cache)
        self.assertIn("pack3", manager._vocabulary_cache)
        self.assertNotIn("pack2", manager._vocabulary_cache, "LRU should evict pack2")

        # Ensure order reflects MRU at end
        # _cache_order keeps keys in access order if cache_size is limited
        self.assertEqual(manager._cache_order[-1], "pack3")

    def test_edge_mode_mini_pack_merge(self):
        # Prepare two different packs for cross-language pair
        _write_pack(self.vocab_dir, "latin", languages=["en"], tokens={f"tok{i}": i for i in range(7000)})
        _write_pack(self.vocab_dir, "cjk", languages=["zh"], tokens={f"汉字{i}": i for i in range(7000)})

        mapping = {"en": "latin", "zh": "cjk"}
        cfg = self._make_config(mapping)

        manager = UnifiedVocabularyManager(config=cfg, vocab_dir=str(self.vocab_dir), mode=VocabularyMode.EDGE, cache_size=3)

        mini = manager.get_vocab_for_pair("en", "zh")
        self.assertEqual(mini.name, "en-zh")
        self.assertEqual(mini.mode, VocabularyMode.EDGE)
        self.assertIn("en", mini.languages)
        self.assertIn("zh", mini.languages)

        # The merged pack should be limited (<= 5000 from each pack)
        self.assertLessEqual(len(mini.tokens), 10000)

        # Tokenization should produce a sequence wrapped by special tokens without errors
        ids = manager.tokenize("hello world", language="en", pack=mini)
        self.assertGreaterEqual(len(ids), 2)  # at least start and end


if __name__ == "__main__":
    unittest.main()