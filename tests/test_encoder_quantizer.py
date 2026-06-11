"""
Tests for training.encoder_quantizer.

Relies on conftest.py for all heavy dependency mocks (torch, numpy, etc.).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.training.quantization.encoder import EncoderQuantizer, QualityPreservingQuantizer


class TestEncoderQuantizerInit(unittest.TestCase):
    """Test EncoderQuantizer initialization."""

    def test_default_init(self):
        quantizer = EncoderQuantizer()
        self.assertIsNotNone(quantizer.config)
        self.assertIsNone(quantizer.calibration_data)
        self.assertIsNotNone(quantizer.quality_comparator)
        self.assertIsNotNone(quantizer.profiler)

    def test_init_with_config(self):
        mock_config = MagicMock()
        mock_config.calibration_samples = 500
        mock_config.backend = 'qnnpack'
        quantizer = EncoderQuantizer(config=mock_config)
        self.assertEqual(quantizer.config.calibration_samples, 500)
        self.assertEqual(quantizer.config.backend, 'qnnpack')


class TestEncoderQuantizerMethods(unittest.TestCase):
    """Test individual quantization methods in isolation."""

    def setUp(self):
        self.quantizer = EncoderQuantizer()
        self.mock_model = MagicMock()
        self.mock_model.state_dict.return_value = {}
        self.mock_model.eval = MagicMock()
        self.mock_model.parameters.return_value = iter([])
        self.mock_model.named_modules.return_value = []
        self.mock_model.half = MagicMock()
        self.mock_model.float = MagicMock()
        self.mock_model.weight = MagicMock()
        self.mock_model.weight.data = MagicMock()

    def test_quantize_dynamic_modern(self):
        import torch
        torch.ao.quantization.quantize_dynamic.reset_mock()
        output_path, metrics = self.quantizer.quantize_dynamic_modern(
            self.mock_model, '/tmp/test_int8.pt'
        )
        torch.ao.quantization.quantize_dynamic.assert_called_once()
        self.assertEqual(output_path, '/tmp/test_int8.pt')
        self.assertIsNone(metrics)

    def test_quantize_dynamic_modern_with_test_data(self):
        self.quantizer.profiler.profile_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.memory_mb = 100.0
        mock_metrics.bleu_score = 0.85
        self.quantizer.profiler.profile_model.return_value = mock_metrics

        output_path, metrics = self.quantizer.quantize_dynamic_modern(
            self.mock_model, '/tmp/test_int8.pt', test_data_path='/tmp/test_data.pt'
        )
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.memory_mb, 100.0)

    def test_quantize_dynamic_modern_with_original_metrics(self):
        self.quantizer.profiler.profile_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.memory_mb = 50.0
        mock_metrics.bleu_score = 0.90
        self.quantizer.profiler.profile_model.return_value = mock_metrics

        original_metrics = MagicMock()
        original_metrics.memory_mb = 200.0
        original_metrics.bleu_score = 0.95

        output_path, metrics = self.quantizer.quantize_dynamic_modern(
            self.mock_model, '/tmp/test_int8.pt',
            test_data_path='/tmp/test_data.pt',
            original_metrics=original_metrics
        )
        self.assertEqual(metrics.compression_ratio, 200.0 / 50.0)

    def test_convert_to_fp16(self):
        self.quantizer._clone_model = MagicMock(return_value=self.mock_model)
        self.mock_model.modules.return_value = []
        self.mock_model.state_dict.return_value = {}

        output_path, metrics = self.quantizer.convert_to_fp16(
            self.mock_model, '/tmp/test_fp16.pt'
        )
        self.mock_model.half.assert_called_once()
        self.assertEqual(output_path, '/tmp/test_fp16.pt')
        self.assertIsNone(metrics)

    def test_convert_to_fp16_with_test_data(self):
        self.quantizer._clone_model = MagicMock(return_value=self.mock_model)
        self.mock_model.modules.return_value = []
        self.mock_model.state_dict.return_value = {}

        self.quantizer.profiler.profile_model = MagicMock()
        self.quantizer.profiler.profile_model.return_value = MagicMock()

        output_path, metrics = self.quantizer.convert_to_fp16(
            self.mock_model, '/tmp/test_fp16.pt', test_data_path='/tmp/test.pt'
        )
        self.assertIsNotNone(metrics)
        self.mock_model.half.assert_called_once()

    def test_quantize_static_fx(self):
        self.quantizer._clone_model = MagicMock(return_value=self.mock_model)
        self.mock_model.state_dict.return_value = {}
        import torch
        torch.load.return_value = [{'input_ids': MagicMock()}]
        torch.ao.quantization.get_default_qconfig.return_value = MagicMock()
        mock_prepared = MagicMock()
        mock_prepared.eval = MagicMock()
        torch.ao.quantization.prepare_fx.return_value = mock_prepared
        torch.ao.quantization.convert_fx.return_value = self.mock_model

        output_path, metrics = self.quantizer.quantize_static_fx(
            self.mock_model, '/tmp/calib_data.pt', '/tmp/test_static_int8.pt'
        )
        self.assertEqual(output_path, '/tmp/test_static_int8.pt')

    def test_create_mixed_precision_model(self):
        self.quantizer._clone_model = MagicMock(return_value=self.mock_model)
        self.mock_model.state_dict.return_value = {}
        self.mock_model.named_modules.return_value = [
            ('embedding_layer', MagicMock()),
            ('encoder.layer1', MagicMock()),
        ]

        output_path, metrics = self.quantizer.create_mixed_precision_model(
            self.mock_model, '/tmp/test_mixed.pt'
        )
        self.assertEqual(output_path, '/tmp/test_mixed.pt')

    def test_get_file_size_mb(self):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(b'x' * 1048576)
            fname = f.name
        try:
            size = self.quantizer._get_file_size_mb(fname)
            self.assertAlmostEqual(size, 1.0, places=2)
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_clone_model(self):
        import copy
        model = MagicMock()
        cloned = self.quantizer._clone_model(model)
        self.assertIsNot(cloned, model)


class TestEncoderQuantizerDistributed(unittest.TestCase):
    """Test distributed quantization methods."""

    def setUp(self):
        self.quantizer = EncoderQuantizer()
        self.mock_model = MagicMock()

    def test_quantize_distributed_world_size_1(self):
        result = self.quantizer.quantize_distributed(self.mock_model, world_size=1)
        self.assertIs(result, self.mock_model)

    def test_quantize_distributed_rank_0(self):
        import torch
        torch.distributed.is_initialized.return_value = True
        torch.distributed.get_rank.return_value = 0
        result = self.quantizer.quantize_distributed(self.mock_model, world_size=2)
        self.assertIs(result, self.mock_model)

    def test_quantize_distributed_rank_nonzero(self):
        import torch
        torch.distributed.is_initialized.return_value = True
        torch.distributed.get_rank.return_value = 1
        result = self.quantizer.quantize_distributed(self.mock_model, world_size=2)
        self.assertIs(result, self.mock_model)


class TestEncoderQuantizerReport(unittest.TestCase):
    """Test quantization report generation."""

    def setUp(self):
        self.quantizer = EncoderQuantizer()

    def test_generate_comparison_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / 'model.pt')
            Path(model_path).touch()

            from pipeline.training.quantization.common import QualityMetrics
            mock_metrics = QualityMetrics(
                latency_ms=10.5, memory_mb=250.0, bleu_score=0.92,
                accuracy=0.90, perplexity=8.0, compression_ratio=2.0,
            )

            fp32_metrics = QualityMetrics(
                latency_ms=15.0, memory_mb=500.0, bleu_score=0.95,
                accuracy=0.93, perplexity=6.0, compression_ratio=1.0,
            )

            results = {
                'fp32': {'size_mb': 500.0, 'metrics': fp32_metrics},
                'int8': {'size_mb': 125.0, 'metrics': mock_metrics},
            }

            self.quantizer._generate_comparison_report(results, model_path)

            report_file = Path(tmpdir) / 'quantization_report.json'
            self.assertTrue(report_file.exists())
            data = json.loads(report_file.read_text())
            self.assertIn('versions', data)
            self.assertIn('int8', data['versions'])

    def test_generate_comparison_report_no_fp32(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / 'model.pt')
            Path(model_path).touch()

            from pipeline.training.quantization.common import QualityMetrics
            mock_metrics = QualityMetrics(
                latency_ms=10.5, memory_mb=125.0, bleu_score=0.85,
                accuracy=0.80, perplexity=12.0, compression_ratio=4.0,
            )

            results = {
                'int8': {'size_mb': 125.0, 'metrics': mock_metrics},
            }

            self.quantizer._generate_comparison_report(results, model_path)
            report_file = Path(tmpdir) / 'quantization_report.json'
            self.assertTrue(report_file.exists())


class TestQualityPreservingQuantizer(unittest.TestCase):
    """Test QualityPreservingQuantizer."""

    def setUp(self):
        self.qpq = QualityPreservingQuantizer()
        self.mock_model = MagicMock()
        self.mock_model.state_dict = MagicMock(return_value={})
        self.mock_model.parameters.return_value = iter([])
        self.mock_model.modules.return_value = []

    def test_quantize_with_calibration(self):
        import torch
        mock_prepared = MagicMock()
        mock_prepared.eval = MagicMock()
        mock_quantized = MagicMock()

        torch.quantization.QConfig = MagicMock()
        torch.quantization.MinMaxObserver = MagicMock()
        torch.quantization.prepare.return_value = mock_prepared
        torch.quantization.convert.return_value = mock_quantized

        self.qpq._evaluate_quality = MagicMock(return_value=0.98)

        result = self.qpq.quantize_with_calibration(
            self.mock_model, [MagicMock()], target_quality=0.97
        )
        self.assertIs(result, mock_quantized)

    def test_evaluate_quality(self):
        import torch
        orig = MagicMock()
        quant = MagicMock()
        orig.eval = MagicMock()
        quant.eval = MagicMock()

        mock_output = MagicMock()
        mock_output.flatten.return_value = MagicMock()
        orig.return_value = mock_output
        quant.return_value = mock_output

        torch.nn.functional.cosine_similarity.return_value.item.return_value = 0.95

        test_data = [MagicMock() for _ in range(5)]
        score = self.qpq._evaluate_quality(orig, quant, test_data)
        self.assertAlmostEqual(score, 0.95)


class TestCreateDeploymentVersions(unittest.TestCase):
    """Test the main create_deployment_versions pipeline."""

    def setUp(self):
        self.quantizer = EncoderQuantizer()
        self.mock_model = MagicMock()
        self.mock_model.eval = MagicMock()
        self.mock_model.state_dict.return_value = {}
        self.mock_model.half = MagicMock()
        self.mock_model.named_modules.return_value = []
        self.mock_model.parameters.return_value = iter([])

    @patch.object(EncoderQuantizer, '_get_file_size_mb', return_value=100.0)
    @patch.object(EncoderQuantizer, 'quantize_dynamic_modern')
    @patch.object(EncoderQuantizer, 'convert_to_fp16')
    @patch.object(EncoderQuantizer, 'create_mixed_precision_model')
    @patch.object(EncoderQuantizer, '_generate_comparison_report')
    def test_create_deployment_versions(
        self, mock_report, mock_mixed, mock_fp16, mock_int8, mock_size
    ):
        import torch
        torch.load.return_value = self.mock_model
        mock_metrics = MagicMock()
        mock_metrics.memory_mb = 200.0
        mock_metrics.latency_ms = 15.0
        mock_metrics.bleu_score = 0.95
        mock_metrics.accuracy = 0.94
        mock_metrics.perplexity = 5.0
        mock_metrics.compression_ratio = 1.0
        self.quantizer.profiler.profile_model = MagicMock(return_value=mock_metrics)

        mock_int8.return_value = ('/tmp/model_int8.pt', MagicMock())
        mock_fp16.return_value = ('/tmp/model_fp16.pt', MagicMock())
        mock_mixed.return_value = ('/tmp/model_mixed.pt', MagicMock())

        results = self.quantizer.create_deployment_versions(
            '/tmp/model.pt', test_data_path='/tmp/test.pt'
        )

        self.assertIn('int8', results)
        self.assertIn('fp16', results)
        self.assertIn('mixed_precision', results)
        self.assertNotIn('static_int8', results)
        self.assertIn('fp32', results)
        mock_report.assert_called_once()

    @patch.object(EncoderQuantizer, '_get_file_size_mb', return_value=100.0)
    @patch.object(EncoderQuantizer, 'quantize_dynamic_modern')
    @patch.object(EncoderQuantizer, 'convert_to_fp16')
    @patch.object(EncoderQuantizer, 'quantize_static_fx')
    @patch.object(EncoderQuantizer, 'create_mixed_precision_model')
    @patch.object(EncoderQuantizer, '_generate_comparison_report')
    def test_create_deployment_versions_with_calibration(
        self, mock_report, mock_mixed, mock_static, mock_fp16, mock_int8, mock_size
    ):
        import torch
        torch.load.return_value = self.mock_model
        mock_metrics = MagicMock()
        mock_metrics.memory_mb = 200.0
        mock_metrics.latency_ms = 15.0
        mock_metrics.bleu_score = 0.95
        mock_metrics.accuracy = 0.94
        mock_metrics.perplexity = 5.0
        mock_metrics.compression_ratio = 1.0
        self.quantizer.profiler.profile_model = MagicMock(return_value=mock_metrics)

        mock_int8.return_value = ('/tmp/model_int8.pt', MagicMock())
        mock_fp16.return_value = ('/tmp/model_fp16.pt', MagicMock())
        mock_static.return_value = ('/tmp/model_static_int8.pt', MagicMock())
        mock_mixed.return_value = ('/tmp/model_mixed.pt', MagicMock())

        results = self.quantizer.create_deployment_versions(
            '/tmp/model.pt',
            calibration_data_path='/tmp/calib.pt',
            test_data_path='/tmp/test.pt',
        )

        self.assertIn('static_int8', results)
        mock_static.assert_called_once()
