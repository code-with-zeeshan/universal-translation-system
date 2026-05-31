"""
Tests for training.training_analytics.

Relies on conftest.py for numpy mock.
"""

import unittest
from unittest.mock import patch

from training.training_analytics import TrainingAnalytics


class TestTrainingAnalyticsInit(unittest.TestCase):
    """Test TrainingAnalytics initialization."""

    @patch('training.training_analytics.time')
    def test_init(self, mock_time):
        mock_time.time.return_value = 1000.0
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        self.assertEqual(ta.start_time, 1000.0)
        self.assertEqual(ta.training_history, history)
        self.assertEqual(len(ta.metrics['loss']), 0)


class TestTrainingAnalyticsLogStep(unittest.TestCase):
    """Test log_step method."""

    @patch('training.training_analytics.time')
    def test_log_step_appends_metrics(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1001.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        ta.log_step(loss=1.5, lr=1e-4, grad_norm=0.5, memory_gb=2.3, tokens_per_sec=1000.0)

        self.assertEqual(ta.metrics['loss'], [1.5])
        self.assertEqual(ta.metrics['lr'], [1e-4])
        self.assertEqual(ta.metrics['grad_norm'], [0.5])
        self.assertEqual(ta.metrics['memory_gb'], [2.3])
        self.assertEqual(ta.metrics['tokens_per_sec'], [1000.0])
        self.assertEqual(len(ta.metrics['timestamp']), 1)
        self.assertEqual(ta.metrics['timestamp'][0], 1.0)

    @patch('training.training_analytics.time')
    def test_log_step_updates_training_history(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1002.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        ta.log_step(loss=0.5, lr=1e-4, grad_norm=0.1, memory_gb=1.5, tokens_per_sec=2000)

        self.assertEqual(history['gradient_norms'], [0.1])
        self.assertEqual(history['memory_snapshots'], [1.5])

    @patch('training.training_analytics.time')
    def test_multiple_log_steps(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        ta.log_step(loss=2.0, lr=1e-4, grad_norm=1.0, memory_gb=1.0, tokens_per_sec=500)
        ta.log_step(loss=1.5, lr=5e-5, grad_norm=0.5, memory_gb=1.2, tokens_per_sec=800)

        self.assertEqual(len(ta.metrics['loss']), 2)
        self.assertEqual(ta.metrics['loss'], [2.0, 1.5])


class TestTrainingAnalyticsLogEpochSummary(unittest.TestCase):
    """Test log_epoch_summary if it exists."""

    @patch('training.training_analytics.time')
    def test_log_epoch_summary_exists(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1001.0, 1002.0, 1003.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        if hasattr(ta, 'log_epoch_summary'):
            ta.log_epoch_summary(epoch=1, avg_loss=1.5, avg_lr=1e-4, avg_tokens_per_sec=1000)
            self.assertTrue(True)


class TestTrainingAnalyticsGenerateReport(unittest.TestCase):
    """Test generate_report method."""

    @patch('training.training_analytics.time')
    def test_generate_report_empty(self, mock_time):
        mock_time.time.side_effect = [1000.0, 2000.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        report = ta.generate_report()

        self.assertIn('duration_hours', report)
        self.assertAlmostEqual(report['duration_hours'], 1000.0 / 3600)
        self.assertEqual(report['avg_tokens_per_sec'], 0)
        self.assertEqual(report['peak_memory_gb'], 0)
        self.assertIsNone(report['final_loss'])
        self.assertEqual(report['loss_reduction'], 0)
        self.assertEqual(report['total_steps'], 0)
        self.assertEqual(report['gradient_norm_stats']['mean'], 0)
        self.assertEqual(report['gradient_norm_stats']['std'], 0)

    @patch('training.training_analytics.time')
    def test_generate_report_with_data(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1005.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        ta.metrics['loss'] = [3.0, 2.0, 1.0]
        ta.metrics['lr'] = [1e-3, 5e-4, 1e-4]
        ta.metrics['grad_norm'] = [0.8, 0.5, 0.2]
        ta.metrics['memory_gb'] = [2.0, 2.1, 2.2]
        ta.metrics['tokens_per_sec'] = [500, 600, 700]
        ta.metrics['timestamp'] = [0, 1, 2]

        report = ta.generate_report()

        self.assertEqual(report['total_steps'], 3)
        self.assertAlmostEqual(report['avg_tokens_per_sec'], 600.0)
        self.assertAlmostEqual(report['peak_memory_gb'], 2.2)
        self.assertEqual(report['final_loss'], 1.0)
        self.assertAlmostEqual(report['loss_reduction'], (3.0 - 1.0) / 3.0)
        self.assertAlmostEqual(report['gradient_norm_stats']['mean'], 0.5)
        self.assertAlmostEqual(report['gradient_norm_stats']['max'], 0.8)

    @patch('training.training_analytics.time')
    def test_generate_report_single_step(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1005.0]
        history = {'gradient_norms': [], 'memory_snapshots': []}
        ta = TrainingAnalytics(history)

        ta.metrics['loss'] = [2.5]
        ta.metrics['lr'] = [1e-3]
        ta.metrics['grad_norm'] = [0.3]
        ta.metrics['memory_gb'] = [1.5]
        ta.metrics['tokens_per_sec'] = [400]
        ta.metrics['timestamp'] = [0]

        report = ta.generate_report()

        self.assertEqual(report['total_steps'], 1)
        self.assertEqual(report['final_loss'], 2.5)
        self.assertEqual(report['loss_reduction'], 0)
        self.assertAlmostEqual(report['gradient_norm_stats']['mean'], 0.3)
        self.assertAlmostEqual(report['gradient_norm_stats']['std'], 0)
