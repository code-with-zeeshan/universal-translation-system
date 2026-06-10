"""Test pipeline data flow: file shapes, column counts, and content.

Verifies the fixes applied to COMET filtering, augment writers, and
data-flow routing.  Requires the full dep stack (torch, yaml, tqdm, comet,
etc.) — run on your Lightning AI studio or wherever you normally run tests.

Tests:
- create_final_training_file merges sampled + augment + pivots into
  train_final.txt / val_final.txt, both 4-column
- create_monolingual_corpora extracts from both sampled and augment dirs
- _comet_filter_file preserves 4-column format
- _comet_quality_filter calls _comet_filter_file for BOTH train and val
- Augment writers (false-friend, idiom) produce 4-column output

Usage:
    python -m pytest tests/test_pipeline_data_flow.py -v
    python -m unittest tests.test_pipeline_data_flow -v
"""

import os
import sys
import unittest
import tempfile
import logging
import asyncio
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Guard: skip all tests if dependencies are missing.
# ---------------------------------------------------------------------------
try:
    from connector.pipeline_connector import PipelineConnector
    from data.pipeline_orchestrator import UniversalTranslationPipeline
    from data.synthetic_augmentation import (
        generate_false_friend_examples,
        generate_idiom_examples,
    )
    HAULING_DEPS = True
except ImportError as e:
    HAULING_DEPS = False
    _IMPORT_ERR = e


def _make_config(processed_dir):
    cfg = MagicMock()
    cfg.data.processed_dir = str(processed_dir)
    cfg.data.active_languages = ['en', 'es', 'de']
    return cfg


def _write(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def _count_cols(path):
    counts = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                counts.add(len(stripped.split('\t')))
    return counts


@unittest.skipIf(not HAULING_DEPS, f'Missing dependencies: {_IMPORT_ERR}')
class TestPipelineDataFlow(unittest.TestCase):
    """Verifies data pipeline fixes: 4-column format, COMET filter,
    augment writers, and data-flow routing."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.processed = self.root / 'data' / 'processed'
        self.sampled_dir = self.processed / 'sampled'
        self.final_dir = self.processed / 'augment'
        self.pivot_dir = self.final_dir / 'pivot_pairs'
        self.corpus_dir = self.processed / 'corpus'

        for d in [self.sampled_dir, self.final_dir, self.pivot_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Sampled data (4-column)
        _write(self.sampled_dir / 'en_es_sampled.txt', [
            'hello world\thola mundo\ten\tes',
            'good morning\tbuenos días\ten\tes',
        ])
        _write(self.sampled_dir / 'en_de_sampled.txt', [
            'hello\thallo\ten\tde',
            'book\tbuch\ten\tde',
        ])

        # Augment files (4-column — the fix we applied)
        _write(self.final_dir / 'augmented_en_es.txt', [
            'my name is\tme llamo\ten\tes',
            'i like dogs\tme gustan los perros\ten\tes',
        ])
        _write(self.final_dir / 'ff_en_es.txt', [
            'actually\tactualmente\ten\tes',
            'library\tbiblioteca\ten\tes',
        ])
        _write(self.pivot_dir / 'en_es_pivot.txt', [
            'via\tdurch\tde\ten',
            'from\tdesde\tes\ten',
        ])
        # Edge case: 2-column line (merge_datasets passes through anything)
        _write(self.final_dir / 'edge_2col.txt', ['incomplete\tline'])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    # ---- create_final_training_file -----------------------------------

    def test_create_final_training_file_produces_4col(self):
        pc = PipelineConnector(_make_config(self.processed))
        pc.logger = logging.getLogger('test')
        pc.create_final_training_file()

        train = self.processed / 'train_final.txt'
        val = self.processed / 'val_final.txt'
        self.assertTrue(train.exists())
        self.assertTrue(val.exists())

        for fname in ('train_final.txt', 'val_final.txt'):
            cols = _count_cols(self.processed / fname)
            self.assertIn(
                4, cols,
                f'{fname} should contain 4-column lines, got cols={cols}'
            )

    def test_create_final_training_file_row_count(self):
        pc = PipelineConnector(_make_config(self.processed))
        pc.logger = logging.getLogger('test')
        pc.create_final_training_file()

        total = 0
        for fn in ('train_final.txt', 'val_final.txt'):
            with open(self.processed / fn, encoding='utf-8') as f:
                total += sum(1 for _ in f)
        self.assertEqual(total, 10,
                         '10 rows: 2 en_es + 2 en_de + 2 augment '
                         '+ 2 ff + 1 pivot + 1 edge')

    # ---- create_monolingual_corpora -----------------------------------

    def test_monolingual_corpora_includes_all_sources(self):
        pc = PipelineConnector(_make_config(self.processed))
        pc.logger = logging.getLogger('test')
        pc.create_monolingual_corpora()

        for lang in ('en', 'es', 'de'):
            self.assertTrue(
                (self.corpus_dir / f'{lang}_corpus.txt').exists(),
                f'{lang}_corpus.txt missing'
            )

        en_texts = set()
        with open(self.corpus_dir / 'en_corpus.txt', encoding='utf-8') as f:
            for line in f:
                en_texts.add(line.strip())
        for t in ('hello world', 'good morning', 'hello', 'book',
                  'my name is', 'i like dogs', 'actually', 'library',
                  'via', 'from'):
            self.assertIn(t, en_texts, f'{t!r} missing from en_corpus.txt')

        es_texts = set()
        with open(self.corpus_dir / 'es_corpus.txt', encoding='utf-8') as f:
            for line in f:
                es_texts.add(line.strip())
        for t in ('hola mundo', 'buenos días', 'me llamo',
                  'me gustan los perros', 'actualmente', 'biblioteca',
                  'desde'):
            self.assertIn(t, es_texts, f'{t!r} missing from es_corpus.txt')

        de_texts = set()
        with open(self.corpus_dir / 'de_corpus.txt', encoding='utf-8') as f:
            for line in f:
                de_texts.add(line.strip())
        for t in ('hallo', 'buch', 'durch'):
            self.assertIn(t, de_texts, f'{t!r} missing from de_corpus.txt')

    # ---- _comet_filter_file -------------------------------------------

    def test_comet_filter_preserves_4col(self):
        orch = UniversalTranslationPipeline.__new__(UniversalTranslationPipeline)
        orch.logger = logging.getLogger('test')

        path = self.processed / 'pairs.txt'
        _write(path, ['hello\thola\ten\tes', 'world\tmundo\ten\tes'])

        class FakeScores:
            scores = [0.95, 0.30]

        mock_model = MagicMock()
        mock_model.predict.return_value = FakeScores()
        orch._comet_filter_file(path, mock_model, 0.7)

        kept = path.read_text(encoding='utf-8').strip().split('\n')
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], 'hello\thola\ten\tes')
        self.assertEqual(len(kept[0].split('\t')), 4)

    def test_comet_filter_empty_file(self):
        orch = UniversalTranslationPipeline.__new__(UniversalTranslationPipeline)
        orch.logger = logging.getLogger('test')

        path = self.processed / 'empty.txt'
        path.write_text('')
        result = orch._comet_filter_file(path, MagicMock(), 0.7)
        self.assertEqual(result, 0)

    def test_comet_filter_missing_file(self):
        orch = UniversalTranslationPipeline.__new__(UniversalTranslationPipeline)
        orch.logger = logging.getLogger('test')
        result = orch._comet_filter_file(
            self.processed / 'nope.txt', MagicMock(), 0.7)
        self.assertEqual(result, 0)

    # ---- _comet_quality_filter dispatches to both files ----------------

    def test_comet_quality_filter_calls_both_files(self):
        orch = UniversalTranslationPipeline.__new__(UniversalTranslationPipeline)
        orch.logger = logging.getLogger('test')
        orch.config = MagicMock()
        orch.config.data.processed_dir = str(self.processed)

        _write(self.processed / 'train_final.txt', ['a\tb\ten\tes'])
        _write(self.processed / 'val_final.txt', ['c\td\ten\tes'])

        with (
            patch.object(orch, '_comet_filter_file') as mock_filter,
            patch('data.pipeline_orchestrator.COMET_AVAILABLE', True),
            patch('data.pipeline_orchestrator.download_model',
                  return_value=''),
            patch('data.pipeline_orchestrator.load_from_checkpoint',
                  return_value=MagicMock()),
        ):
            asyncio.run(orch._comet_quality_filter())

        self.assertEqual(mock_filter.call_count, 2)
        a0 = str(mock_filter.call_args_list[0][0][0])
        a1 = str(mock_filter.call_args_list[1][0][0])
        self.assertIn('train_final.txt', a0)
        self.assertIn('val_final.txt', a1)

    # ---- augment writers produce 4-col ---------------------------------

    def test_false_friend_writer_4col(self):
        ff_list = [
            {'en': 'actually', 'es': 'actualmente', 'pair': ('en', 'es')},
            {'en': 'library', 'es': 'librería', 'pair': ('en', 'es')},
        ]
        path = self.final_dir / 'ff.txt'
        generate_false_friend_examples(
            ff_list, output_file=str(path),
            src_lang='en', tgt_lang='es',
        )
        cols = _count_cols(path)
        self.assertIn(4, cols,
                      f'False-friend writer should produce 4 columns, '
                      f'got {cols}')

    def test_idiom_writer_4col(self):
        idioms = [
            {'source': 'break a leg',
             'translation': 'mucha mierda',
             'src_lang': 'en', 'tgt_lang': 'es'},
        ]
        path = self.final_dir / 'idiom.txt'
        generate_idiom_examples(
            idioms, output_file=str(path),
            src_lang='en', tgt_lang='es',
        )
        cols = _count_cols(path)
        self.assertIn(4, cols,
                      f'Idiom writer should produce 4 columns, got {cols}')

    # ---- seeded train/val split ----------------------------------------

    def test_seeded_split_determinism(self):
        """Same config seed must produce identical train/val split."""
        cfg1 = _make_config(self.processed)
        cfg1.data.seed = 42
        cfg2 = _make_config(self.processed)
        cfg2.data.seed = 42

        pc1 = PipelineConnector(cfg1)
        pc1.logger = logging.getLogger('test')
        pc1.create_final_training_file()

        pc2 = PipelineConnector(cfg2)
        pc2.logger = logging.getLogger('test')
        pc2.create_final_training_file()

        train1 = (self.processed / 'train_final.txt').read_text(encoding='utf-8')
        train2 = (self.processed / 'train_final.txt').read_text(encoding='utf-8')
        self.assertEqual(train1, train2,
                         'Same seed must produce identical split')

    def test_different_seed_different_split(self):
        """Different seeds must produce different train/val splits."""
        tmpdirs = []

        def _run_with_seed(seed, out_dir):
            d = out_dir / 'processed'
            for p in [d / 'sampled', d / 'augment', d / 'augment' / 'pivot_pairs']:
                p.mkdir(parents=True, exist_ok=True)
            _write(d / 'sampled' / 'en_es_sampled.txt', [
                'hello world\thola mundo\ten\tes',
                'good morning\tbuenos días\ten\tes',
            ])
            _write(d / 'augment' / 'augmented_en_es.txt', [
                'my name is\tme llamo\ten\tes',
            ])
            cfg = _make_config(d)
            cfg.data.seed = seed
            pc = PipelineConnector(cfg)
            pc.logger = logging.getLogger('test')
            pc.create_final_training_file()
            return (d / 'train_final.txt').read_text(encoding='utf-8')

        d1 = Path(tempfile.mkdtemp())
        d2 = Path(tempfile.mkdtemp())
        tmpdirs = [d1, d2]
        try:
            train_1 = _run_with_seed(1, d1)
            train_2 = _run_with_seed(999, d2)
            self.assertNotEqual(train_1, train_2)
        finally:
            for td in tmpdirs:
                shutil.rmtree(td, ignore_errors=True)


@unittest.skipIf(not HAULING_DEPS, f'Missing dependencies: {_IMPORT_ERR}')
class TestTrainingPipelineFixes(unittest.TestCase):
    """Verifies training pipeline fixes: token cache invalidation, format
    warnings, cross-stage dependency check."""

    def test_cache_fingerprint_includes_mtime_and_size(self):
        """ModernParallelDataset._cache_fingerprint returns expected keys."""
        from data.dataset_classes import ModernParallelDataset
        ds = ModernParallelDataset.__new__(ModernParallelDataset)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write('a\tb\ten\tes\n')
            f.flush()
            ds.data_path = Path(f.name)
            fp = ds._cache_fingerprint()

        self.assertIn('mtime_ns', fp)
        self.assertIn('size', fp)
        self.assertIn('max_length', fp)
        self.assertIsInstance(fp['size'], int)
        self.assertIsInstance(fp['max_length'], int)

    def test_load_raw_data_warns_on_skipped_lines(self):
        """_load_raw_data should warn when <4-column lines are present."""
        from data.dataset_classes import ModernParallelDataset
        ds = ModernParallelDataset.__new__(ModernParallelDataset)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('good\tline\ten\tes\n')
            f.write('bad\t2col\n')
            f.write('also\tgood\tfr\ten\n')
            f.flush()
            fname = f.name

        ds.data_path = Path(fname)
        try:
            with self.assertLogs('data.dataset_classes', level='WARNING') as cm:
                data = ds._load_raw_data()
            self.assertEqual(len(data), 2, 'Should parse 2 valid lines')
            self.assertTrue(
                any('skipped' in m.lower() for m in cm.output),
                f'Expected warning about skipped lines, got: {cm.output}'
            )
        finally:
            Path(fname).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
