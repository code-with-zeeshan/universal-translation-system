"""
Tests for training.training_utils.

Relies on conftest.py for all heavy dependency mocks (torch, wandb, numpy, etc.).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.training.utils import (
    BaseTrainer,
    check_convergence,
    find_convergence_step,
    create_training_report,
    calculate_gradient_norm,
    create_optimizer_with_param_groups,
    get_adaptive_gradient_clipping_value,
    get_learning_rate_schedule,
    get_training_diagnostics,
    save_training_state,
)


class TestBaseTrainer(unittest.TestCase):
    """Test BaseTrainer abstract class."""

    def test_abstract_class_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseTrainer(
                encoder=MagicMock(), decoder=MagicMock(),
                train_data_path='/tmp/train.pt', val_data_path='/tmp/val.pt',
                config=MagicMock(), experiment_name='test',
            )


class TestBaseTrainerConcrete(unittest.TestCase):
    """Test BaseTrainer with a concrete subclass."""

    def setUp(self):
        class ConcreteTrainer(BaseTrainer):
            def _setup_models(self):
                pass
            def train(self, num_epochs):
                pass
            def _train_epoch(self, epoch, train_loader):
                pass
            def _validate_epoch(self, val_loader):
                pass

        self.mock_config = MagicMock()
        self.mock_config.training = MagicMock()
        self.mock_config.training.checkpoint_dir = '/tmp/checkpoints'
        self.mock_config.monitoring = MagicMock()
        self.mock_config.monitoring.use_wandb = False

        self.trainer = ConcreteTrainer(
            encoder=MagicMock(), decoder=MagicMock(),
            train_data_path='/tmp/train.pt', val_data_path='/tmp/val.pt',
            config=self.mock_config, experiment_name='test_experiment',
        )

    def test_init_sets_attributes(self):
        self.assertIsNotNone(self.trainer.encoder)
        self.assertIsNotNone(self.trainer.decoder)
        self.assertEqual(self.trainer.experiment_name, 'test_experiment')
        self.assertEqual(self.trainer.global_step, 0)
        self.assertEqual(self.trainer.best_val_loss, float('inf'))

    def test_init_device_cpu(self):
        self.assertEqual(str(self.trainer.device), 'cpu')

    def test_init_creates_checkpoint_dir(self):
        self.mock_config.training.checkpoint_dir = '/tmp/checkpoints'
        import pathlib
        self.assertIsInstance(self.trainer.checkpoint_dir, pathlib.Path)
        self.assertTrue(self.trainer.checkpoint_dir.exists())

    def test_init_does_not_init_wandb(self):
        import wandb
        wandb.init.assert_not_called()


class TestBaseTrainerSetupWandb(unittest.TestCase):
    """Test _setup_wandb method."""

    def test_setup_wandb_enabled(self):
        import wandb

        class ConcreteTrainer(BaseTrainer):
            def _setup_models(self):
                pass
            def train(self, num_epochs):
                pass
            def _train_epoch(self, epoch, train_loader):
                pass
            def _validate_epoch(self, val_loader):
                pass

        mock_config = MagicMock()
        mock_config.training = MagicMock()
        mock_config.training.checkpoint_dir = '/tmp/checkpoints'
        mock_config.monitoring = MagicMock()
        mock_config.monitoring.use_wandb = True
        mock_config.dict.return_value = {'test': 'config'}

        trainer = ConcreteTrainer(
            encoder=MagicMock(), decoder=MagicMock(),
            train_data_path='/tmp/train.pt', val_data_path='/tmp/val.pt',
            config=mock_config, experiment_name='test_exp',
        )
        wandb.init.assert_called_once()


class TestCheckConvergence(unittest.TestCase):
    """Test check_convergence function."""

    def test_not_enough_losses(self):
        self.assertFalse(check_convergence([1.0, 1.1, 1.2], window=5, threshold=0.001))

    def test_converged(self):
        losses = [1.0] * 200 + [0.5] * 200
        self.assertTrue(check_convergence(losses, window=100, threshold=0.001))

    def test_not_converged(self):
        losses = list(range(300))
        self.assertFalse(check_convergence(losses, window=100, threshold=0.001))

    def test_custom_window_and_threshold(self):
        losses = [1.0] * 20 + [1.5] * 20
        self.assertTrue(check_convergence(losses, window=10, threshold=0.1))

    def test_edge_exact_threshold(self):
        # losses[-200:-100] = [1.0]*100, losses[-100:] = [1.0005]*100, diff=0.0005
        losses = [1.0] * 300 + [1.0005] * 100
        # threshold=0.0005: diff is 0.0005, which is NOT < 0.0005
        result = check_convergence(losses, window=100, threshold=0.0005)
        diff = abs(sum(losses[-100:]) / 100 - sum(losses[-200:-100]) / 100)
        self.assertAlmostEqual(diff, 0.0005)
        self.assertFalse(result)

    def test_empty_losses(self):
        self.assertFalse(check_convergence([], window=100, threshold=0.001))


class TestFindConvergenceStep(unittest.TestCase):
    """Test find_convergence_step function."""

    def test_returns_step(self):
        losses = [1.0] * 50 + [0.5] * 250
        step = find_convergence_step(losses, window=100, threshold=0.001)
        self.assertIsNotNone(step)
        self.assertGreaterEqual(step, 200)

    def test_no_convergence(self):
        losses = list(range(300))
        step = find_convergence_step(losses, window=100, threshold=0.001)
        self.assertIsNone(step)

    def test_not_enough_data(self):
        losses = [1.0, 2.0]
        step = find_convergence_step(losses, window=10, threshold=0.001)
        self.assertIsNone(step)


class TestCreateTrainingReport(unittest.TestCase):
    """Test create_training_report function."""

    def test_create_report(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            out_path = f.name

        try:
            history = {'loss': [3.0, 2.0, 1.0], 'step_times': [0.1, 0.2, 0.3]}
            report = create_training_report(history, output_path=out_path)

            self.assertEqual(report['final_metrics']['final_loss'], 1.0)
            self.assertEqual(report['final_metrics']['best_loss'], 1.0)
            self.assertEqual(report['final_metrics']['total_steps'], 3)
            self.assertAlmostEqual(report['performance']['avg_step_time'], 0.2)
            self.assertAlmostEqual(report['performance']['total_training_time'], 0.6)
            self.assertEqual(report['full_history'], history)
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_create_report_empty_losses(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            out_path = f.name

        try:
            history = {'loss': [], 'step_times': []}
            report = create_training_report(history, output_path=out_path)

            self.assertIsNone(report['final_metrics']['final_loss'])
            self.assertIsNone(report['final_metrics']['best_loss'])
            self.assertEqual(report['final_metrics']['total_steps'], 0)
            self.assertFalse(report['convergence']['converged'])
        finally:
            Path(out_path).unlink(missing_ok=True)


class TestCalculateGradientNorm(unittest.TestCase):
    """Test calculate_gradient_norm function."""

    def test_single_model_no_gradients(self):
        model = MagicMock()
        model.parameters.return_value = iter([])
        norm = calculate_gradient_norm(model)
        self.assertEqual(norm, 0.0)

    def test_single_model_with_gradients(self):
        import torch
        param = MagicMock()
        param.grad = MagicMock()
        param.grad.data = MagicMock()
        param.grad.data.norm.return_value.item.return_value = 1.0
        param.grad.device = 'cpu'
        param.grad.detach.return_value = param.grad
        torch.norm.return_value.item.return_value = 1.0

        model = MagicMock()
        model.parameters.return_value = iter([param])

        norms = calculate_gradient_norm(model)
        self.assertGreaterEqual(norms, 0.0)

    def test_per_layer(self):
        param = MagicMock()
        param.grad = MagicMock()
        param.grad.data = MagicMock()
        param.grad.data.norm.return_value.item.return_value = 0.5
        param.grad.device = 'cpu'

        model = MagicMock()
        model.named_parameters.return_value = [('layer1.weight', param)]
        model.parameters.return_value = iter([param])

        # Pass as list so isninstance(model, nn.Module) check doesn't fail
        norms = calculate_gradient_norm([model], per_layer=True)
        self.assertIn('model.layer1.weight', norms)
        self.assertEqual(norms['model.layer1.weight'], 0.5)


class TestCreateOptimizer(unittest.TestCase):
    """Test create_optimizer_with_param_groups function."""

    def test_single_model(self):
        param = MagicMock()
        param.requires_grad = True
        param.numel.return_value = 100

        model = MagicMock()
        model.named_parameters.return_value = [('embed.weight', param)]
        model.parameters.return_value = iter([param])

        config = MagicMock()
        config.training = MagicMock()
        config.training.learning_rate = 0.001
        config.training.optimizer_type = 'AdamW'
        config.training.weight_decay = 0.01

        optimizer = create_optimizer_with_param_groups(model, config)
        self.assertIsNotNone(optimizer)

    def test_model_list(self):
        param = MagicMock()
        param.requires_grad = True
        param.numel.return_value = 100

        enc = MagicMock()
        enc.named_parameters.return_value = [('embed.weight', param)]
        enc.parameters.return_value = iter([param])

        dec = MagicMock()
        dec.named_parameters.return_value = [('transformer.weight', param)]
        dec.parameters.return_value = iter([param])

        config = MagicMock()
        config.training = MagicMock()
        config.training.learning_rate = 0.001
        config.training.optimizer_type = 'AdamW'
        config.training.weight_decay = 0.01

        optimizer = create_optimizer_with_param_groups([enc, dec], config)
        self.assertIsNotNone(optimizer)


class TestAdaptiveGradientClipping(unittest.TestCase):
    """Test get_adaptive_gradient_clipping_value function."""

    def test_short_history(self):
        value = get_adaptive_gradient_clipping_value([0.5, 0.6, 0.4])
        self.assertEqual(value, 1.0)

    def test_returns_clipped_value(self):
        value = get_adaptive_gradient_clipping_value(
            [0.5] * 20, percentile=50, min_clip=0.1, max_clip=10.0
        )
        self.assertAlmostEqual(value, 0.5)

    def test_empty_history(self):
        value = get_adaptive_gradient_clipping_value([])
        self.assertEqual(value, 1.0)


class TestGetLearningRateSchedule(unittest.TestCase):
    """Test get_learning_rate_schedule function."""

    def test_extract_lrs(self):
        optimizer = MagicMock()
        optimizer.param_groups = [{'lr': 0.001}, {'lr': 0.0005}]
        lrs = get_learning_rate_schedule(optimizer)
        self.assertEqual(lrs, [0.001, 0.0005])

    def test_empty_groups(self):
        optimizer = MagicMock()
        optimizer.param_groups = []
        lrs = get_learning_rate_schedule(optimizer)
        self.assertEqual(lrs, [])


class TestSaveTrainingState(unittest.TestCase):
    """Test save_training_state function."""

    def test_save_with_torch(self):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            out_path = f.name

        try:
            import torch
            torch.save.reset_mock()
            state = {'epoch': 1, 'loss': 0.5}
            save_training_state(state, out_path, use_safetensors=False)
            torch.save.assert_called_once()
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_save_with_safetensors_fallback(self):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            out_path = f.name

        try:
            import torch
            torch.save.reset_mock()
            state = {'epoch': 1, 'loss': 0.5}
            save_training_state(state, out_path, use_safetensors=True)
            torch.save.assert_called_once()
        finally:
            Path(out_path).unlink(missing_ok=True)


class TestGetTrainingDiagnostics(unittest.TestCase):
    """Test get_training_diagnostics function."""

    def test_basic_diagnostics_keys(self):
        trainer = MagicMock()
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 0.5

        diagnostics = get_training_diagnostics(trainer)
        self.assertIn('timestamp', diagnostics)
        self.assertIn('training_state', diagnostics)
        self.assertIn('model_state', diagnostics)
        self.assertIn('optimizer_state', diagnostics)
        self.assertIn('memory_state', diagnostics)
        self.assertIn('performance_metrics', diagnostics)
        self.assertIn('convergence_indicators', diagnostics)
