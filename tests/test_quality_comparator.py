"""
Tests for training.quality_comparator.

Relies on conftest.py for all heavy dependency mocks (torch, numpy, etc.).
"""

import unittest
from unittest.mock import MagicMock, patch

from training.quality_comparator import QualityComparator


class TestQualityComparatorInit(unittest.TestCase):
    """Test QualityComparator initialization."""

    def test_default_init(self):
        qc = QualityComparator()
        self.assertFalse(qc.quantization_aware)

    def test_enable_quantization_aware_training(self):
        qc = QualityComparator()
        qc.enable_quantization_aware_training()
        self.assertTrue(qc.quantization_aware)


class TestQualityComparatorFakeQuantize(unittest.TestCase):
    """Test fake_quantize method."""

    def test_fake_quantize_disabled(self):
        qc = QualityComparator()
        tensor = MagicMock()
        result = qc.fake_quantize(tensor, num_bits=8)
        self.assertIs(result, tensor)

    def test_fake_quantize_enabled(self):
        with patch('training.quality_comparator.fake_quantize_tensor') as mock_fqt:
            qc = QualityComparator()
            qc.enable_quantization_aware_training()
            tensor = MagicMock()
            mock_fqt.return_value = MagicMock()
            result = qc.fake_quantize(tensor, num_bits=8)
            mock_fqt.assert_called_once_with(tensor, 8)


class TestQualityComparatorCompareModels(unittest.TestCase):
    """Test compare_models method."""

    def setUp(self):
        self.qc = QualityComparator()

    @patch.object(QualityComparator, '_translate')
    @patch.object(QualityComparator, '_get_model_size_mb')
    def test_compare_models_single(self, mock_size, mock_translate):
        mock_translate.return_value = "hello world"
        mock_size.return_value = 100.0

        model = MagicMock()
        results = self.qc.compare_models("hola mundo", "es", "en", {'fp32': model})

        self.assertIn('fp32', results)
        self.assertEqual(results['fp32']['translation'], "hello world")
        self.assertIn('latency_ms', results['fp32'])
        self.assertEqual(results['fp32']['model_size_mb'], 100.0)

    @patch.object(QualityComparator, '_translate')
    @patch.object(QualityComparator, '_get_model_size_mb')
    @patch.object(QualityComparator, '_calculate_similarity')
    def test_compare_models_multi(self, mock_similarity, mock_size, mock_translate):
        mock_translate.side_effect = ["hello world", "hello world"]
        mock_size.side_effect = [500.0, 125.0]
        mock_similarity.return_value = 0.95

        models = {'fp32': MagicMock(), 'int8': MagicMock()}
        results = self.qc.compare_models("hola mundo", "es", "en", models)

        self.assertIn('fp32', results)
        self.assertIn('int8', results)
        self.assertIn('similarity_score', results['int8'])
        self.assertEqual(results['int8']['similarity_score'], 0.95)

    @patch.object(QualityComparator, '_translate')
    @patch.object(QualityComparator, '_get_model_size_mb')
    def test_compare_models_no_reference(self, mock_size, mock_translate):
        mock_translate.side_effect = ["hello", "world"]
        mock_size.return_value = 100.0

        models = {'int8': MagicMock(), 'fp16': MagicMock()}
        results = self.qc.compare_models("test", "es", "en", models)

        self.assertNotIn('similarity_score', results['int8'])
        self.assertNotIn('similarity_score', results['fp16'])


class TestQualityComparatorHelpers(unittest.TestCase):
    """Test helper methods for metric calculation."""

    def setUp(self):
        self.qc = QualityComparator()

    def test_get_model_size_mb(self):
        param = MagicMock()
        param.nelement.return_value = 1000
        param.element_size.return_value = 4
        model = MagicMock()
        model.parameters.return_value = iter([param])

        size = self.qc._get_model_size_mb(model)
        self.assertEqual(size, 1000 * 4 / (1024 * 1024))

    @patch('difflib.SequenceMatcher')
    def test_calculate_similarity(self, mock_sm):
        mock_sm.return_value.ratio.return_value = 0.85
        score = self.qc._calculate_similarity("hello world", "hello world")
        self.assertEqual(score, 0.85)

    def test_calculate_accuracy(self):
        refs = ["hello", "world", "foo"]
        trans = ["hello", "world", "bar"]
        acc = self.qc._calculate_accuracy(refs, trans)
        self.assertAlmostEqual(acc, 2.0 / 3.0)

    @patch('sacrebleu.corpus_bleu')
    def test_calculate_bleu(self, mock_bleu):
        mock_bleu.return_value.score = 42.0
        score = self.qc._calculate_bleu(["ref1", "ref2"], ["hyp1", "hyp2"])
        self.assertEqual(score, 42.0)


class TestQualityComparatorTranslate(unittest.TestCase):
    """Test the _translate method via proper mocking of its dynamic import."""

    def setUp(self):
        self.qc = QualityComparator()

    @patch('vocabulary.unified_vocab_manager.UnifiedVocabularyManager')
    @patch('vocabulary.unified_vocab_manager.VocabularyMode')
    def test_translate_tuple_model(self, mock_mode, mock_vm_cls):
        mock_vm = MagicMock()
        mock_vm_cls.return_value = mock_vm
        mock_vocab_pack = MagicMock()
        mock_vm.get_vocab_for_pair.return_value = mock_vocab_pack
        mock_vocab_pack.special_tokens = {'<s>': 2, '</s>': 3, '<unk>': 1}
        mock_vocab_pack.tokens = {'hello': 10, 'world': 20}

        encoder = MagicMock()
        decoder = MagicMock()
        mock_decoder_out = MagicMock()
        decoder.return_value = mock_decoder_out
        mock_decoder_out.argmax.return_value = MagicMock()
        mock_decoder_out.argmax.return_value.item.side_effect = [10, 20, 3]

        result = self.qc._translate((encoder, decoder), "hello world", "es", "en")
        self.assertIsInstance(result, str)

    @patch('vocabulary.unified_vocab_manager.UnifiedVocabularyManager')
    @patch('vocabulary.unified_vocab_manager.VocabularyMode')
    def test_translate_combined_model(self, mock_mode, mock_vm_cls):
        mock_vm = MagicMock()
        mock_vm_cls.return_value = mock_vm
        mock_vocab_pack = MagicMock()
        mock_vm.get_vocab_for_pair.return_value = mock_vocab_pack
        mock_vocab_pack.special_tokens = {'<s>': 2, '</s>': 3, '<unk>': 1}
        mock_vocab_pack.tokens = {'hello': 10}

        model = MagicMock()
        model.encoder = MagicMock()
        model.decoder = MagicMock()
        mock_decoder_out = MagicMock()
        model.decoder.return_value = mock_decoder_out
        mock_decoder_out.argmax.return_value = MagicMock()
        mock_decoder_out.argmax.return_value.item.side_effect = [10, 3]

        result = self.qc._translate(model, "hello", "es", "en")
        self.assertIsInstance(result, str)

    def test_translate_unknown_model(self):
        model = MagicMock(spec=object)
        result = self.qc._translate(model, "hello", "es", "en")
        self.assertEqual(result, "")


class TestQualityComparatorPerplexity(unittest.TestCase):
    """Test perplexity calculation."""

    def setUp(self):
        self.qc = QualityComparator()

    def test_calculate_perplexity(self):
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = MagicMock()

        import torch
        mock_loss = MagicMock()
        mock_loss.item.side_effect = [2.5, 2.9]
        torch.nn.CrossEntropyLoss.return_value = MagicMock()
        torch.nn.CrossEntropyLoss.return_value.return_value = mock_loss

        data_loader = [
            {'input_ids': MagicMock(), 'labels': MagicMock()},
            {'input_ids': MagicMock(), 'labels': MagicMock()},
        ]

        perplexity = self.qc._calculate_perplexity(mock_model, data_loader)
        self.assertAlmostEqual(perplexity, math.exp(2.7), places=5)

import math
