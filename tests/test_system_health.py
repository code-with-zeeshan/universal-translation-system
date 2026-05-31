"""
Tests for integration.system_health.

Relies on conftest.py for all heavy dependency mocks (psutil, torch, prometheus_client).
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from integration.system_health import SystemHealthMonitor


class TestSystemHealthMonitorInit(unittest.TestCase):
    """Test SystemHealthMonitor initialization."""

    def test_init_with_system(self):
        mock_system = MagicMock()
        monitor = SystemHealthMonitor(mock_system)
        self.assertIs(monitor.system, mock_system)
        self.assertEqual(monitor.health_metrics, {})

    def test_executor_created(self):
        monitor = SystemHealthMonitor(MagicMock())
        self.assertIsNotNone(monitor.executor)


class TestSystemHealthCheckHealth(unittest.TestCase):
    """Test check_health method."""

    def setUp(self):
        self.mock_system = MagicMock()
        self.mock_system.config.data_dir = '/tmp/data'
        self.monitor = SystemHealthMonitor(self.mock_system)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_check_health_returns_dict(self):
        result = self._run(self.monitor.check_health())
        self.assertIn('status', result)
        self.assertIn('timestamp', result)
        self.assertIn('components', result)
        self.assertIn('resources', result)
        self.assertEqual(result['status'], 'healthy')

    def test_check_health_components_present(self):
        result = self._run(self.monitor.check_health())
        expected = ['data_pipeline', 'vocab_manager', 'encoder',
                    'decoder', 'trainer', 'evaluator']
        for comp in expected:
            self.assertIn(comp, result['components'])

    def test_check_health_resources_present(self):
        result = self._run(self.monitor.check_health())
        self.assertIn('cpu_percent', result['resources'])
        self.assertIn('memory_percent', result['resources'])
        self.assertIn('memory_available_gb', result['resources'])

    def test_check_health_gpu_resources(self):
        result = self._run(self.monitor.check_health())
        self.assertIn('gpu', result['resources'])

    def test_check_health_degraded_on_exception(self):
        self.monitor._check_data_pipeline = MagicMock()
        self.monitor._check_data_pipeline.side_effect = Exception("Pipeline error")
        result = self._run(self.monitor.check_health())
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['components']['data_pipeline']['status'], 'error')


class TestSystemHealthCheckDataPipeline(unittest.TestCase):
    """Test _check_data_pipeline method."""

    def setUp(self):
        self.mock_system = MagicMock()
        self.mock_system.config.data_dir = '/tmp/data'
        self.monitor = SystemHealthMonitor(self.mock_system)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_not_initialized(self):
        self.mock_system.data_pipeline = None
        result = self._run(self.monitor._check_data_pipeline())
        self.assertEqual(result['status'], 'not_initialized')

    def test_healthy(self):
        self.mock_system.data_pipeline = MagicMock()
        with patch('integration.system_health.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            result = self._run(self.monitor._check_data_pipeline())
            self.assertEqual(result['status'], 'healthy')


class TestSystemHealthCheckVocabManager(unittest.TestCase):
    """Test _check_vocab_manager method."""

    def setUp(self):
        self.monitor = SystemHealthMonitor(MagicMock())

    def _run(self, coro):
        return asyncio.run(coro)

    def test_not_initialized(self):
        self.monitor.system.vocab_manager = None
        result = self._run(self.monitor._check_vocab_manager())
        self.assertEqual(result['status'], 'not_initialized')

    def test_healthy(self):
        self.monitor.system.vocab_manager = MagicMock()
        self.monitor.system.vocab_manager.get_loaded_versions.return_value = ['v1', 'v2']
        result = self._run(self.monitor._check_vocab_manager())
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['loaded_packs'], 2)

    def test_error(self):
        self.monitor.system.vocab_manager = MagicMock()
        self.monitor.system.vocab_manager.get_loaded_versions.side_effect = Exception("Vocab error")
        result = self._run(self.monitor._check_vocab_manager())
        self.assertEqual(result['status'], 'error')


class TestSystemHealthCheckEncoderDecoder(unittest.TestCase):
    """Test encoder/decoder component checks."""

    def setUp(self):
        self.monitor = SystemHealthMonitor(MagicMock())

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_model(self, training=False):
        model = MagicMock()
        model.parameters.return_value = iter([MagicMock()])
        model.training = training
        return model

    def test_encoder_not_initialized(self):
        self.monitor.system.encoder = None
        result = self._run(self.monitor._check_encoder())
        self.assertEqual(result['status'], 'not_initialized')

    def test_encoder_healthy(self):
        self.monitor.system.encoder = self._make_model(training=False)
        result = self._run(self.monitor._check_encoder())
        self.assertEqual(result['status'], 'healthy')
        self.assertIn('device', result)
        self.assertIn('parameters', result)

    def test_decoder_not_initialized(self):
        self.monitor.system.decoder = None
        result = self._run(self.monitor._check_decoder())
        self.assertEqual(result['status'], 'not_initialized')

    def test_decoder_healthy(self):
        self.monitor.system.decoder = self._make_model(training=True)
        result = self._run(self.monitor._check_decoder())
        self.assertEqual(result['status'], 'healthy')

    def test_trainer_healthy(self):
        self.monitor.system.trainer = MagicMock()
        result = self._run(self.monitor._check_trainer())
        self.assertEqual(result['status'], 'healthy')

    def test_trainer_not_initialized(self):
        self.monitor.system.trainer = None
        result = self._run(self.monitor._check_trainer())
        self.assertEqual(result['status'], 'not_initialized')

    def test_evaluator_healthy(self):
        self.monitor.system.evaluator = MagicMock()
        result = self._run(self.monitor._check_evaluator())
        self.assertEqual(result['status'], 'healthy')


class TestSystemHealthCheckResources(unittest.TestCase):
    """Test _check_resources method."""

    def setUp(self):
        self.monitor = SystemHealthMonitor(MagicMock())

    def _run(self, coro):
        return asyncio.run(coro)

    def test_cpu_memory_metrics(self):
        result = self._run(self.monitor._check_resources())
        self.assertIn('cpu_percent', result)
        self.assertIn('memory_percent', result)
        self.assertIn('memory_available_gb', result)

    def test_gpu_metrics_when_available(self):
        import torch
        torch.cuda.is_available.return_value = True
        result = self._run(self.monitor._check_resources())
        self.assertIn('gpu', result)
        self.assertEqual(result['gpu']['memory_allocated_gb'], 2.0)

    def test_no_gpu_when_unavailable(self):
        import torch
        torch.cuda.is_available.return_value = False
        result = self._run(self.monitor._check_resources())
        self.assertNotIn('gpu', result)

    def test_gpu_utilization_included(self):
        self.monitor._get_gpu_utilization = MagicMock(return_value=75.5)
        result = self._run(self.monitor._check_resources())
        self.assertEqual(result['gpu']['utilization'], 75.5)


class TestSystemHealthGetGpuUtilization(unittest.TestCase):
    """Test _get_gpu_utilization method."""

    def setUp(self):
        self.monitor = SystemHealthMonitor(MagicMock())

    def test_successful(self):
        # conftest.py already mocks nvidia_ml_py3; call the method directly
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit = MagicMock()
        nvml.nvmlDeviceGetHandleByIndex = MagicMock()
        nvml.nvmlDeviceGetUtilizationRates = MagicMock()
        nvml.nvmlDeviceGetUtilizationRates.return_value.gpu = 80

        result = self.monitor._get_gpu_utilization()
        self.assertEqual(result, 80)

    def test_exception_returns_none(self):
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit.side_effect = Exception("NVML error")

        result = self.monitor._get_gpu_utilization()
        self.assertIsNone(result)


class TestSystemHealthValidateConfig(unittest.TestCase):
    """Test validate_configuration method."""

    def setUp(self):
        self.mock_system = MagicMock()
        self.monitor = SystemHealthMonitor(self.mock_system)

    def _make_config(self, **overrides):
        config = MagicMock()
        config.data_dir = overrides.get('data_dir', '/tmp/data')
        config.model_dir = overrides.get('model_dir', '/tmp/models')
        config.vocab_dir = overrides.get('vocab_dir', '/tmp/vocabs')
        config.device = overrides.get('device', 'cpu')
        config.enable_monitoring = overrides.get('enable_monitoring', False)
        config.monitoring_port = overrides.get('monitoring_port', 8000)
        return config

    def test_no_errors(self):
        self.monitor.config = self._make_config()
        with patch('integration.system_health.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            errors = self.monitor.validate_configuration()
            self.assertEqual(errors, [])

    def test_missing_dirs(self):
        self.monitor.config = self._make_config(
            data_dir='/nonexistent/data',
            model_dir='/nonexistent/models',
            vocab_dir='/nonexistent/vocabs',
        )
        with patch('integration.system_health.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            errors = self.monitor.validate_configuration()
            self.assertGreater(len(errors), 0)

    def test_cuda_unavailable(self):
        import torch
        torch.cuda.is_available.return_value = False
        self.monitor.config = self._make_config(device='cuda')
        with patch('integration.system_health.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            errors = self.monitor.validate_configuration()
            self.assertIn("CUDA requested but not available", errors)

    def test_monitoring_port_check(self):
        self.monitor.config = self._make_config(enable_monitoring=True, monitoring_port=8000)
        with patch('integration.system_health.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('socket.socket') as mock_socket:
                errors = self.monitor.validate_configuration()
                mock_socket.assert_called_once()
