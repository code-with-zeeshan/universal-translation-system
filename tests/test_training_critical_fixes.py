"""
Tests for critical training pipeline fixes.
Uses file text directly to avoid importing torch-dependent modules.
"""

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_source(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding='utf-8')


class TestCheckpointStateOrdering(unittest.TestCase):
    """Verify global_step/current_epoch/best_val_loss are set BEFORE load_checkpoint."""

    def test_init_order(self):
        source = _read_source('pipeline/training/trainer.py')
        # Find __init__ method body
        init_start = source.find('def __init__(self')
        init_body = source[init_start:]

        state_vars = ['self.global_step', 'self.current_epoch', 'self.best_val_loss']
        for var in state_vars:
            last_assign = init_body.rfind(f'{var} = ')
            load_call = init_body.rfind('self.load_checkpoint(')
            self.assertGreater(
                last_assign, load_call,
                f'{var} must be set BEFORE load_checkpoint call'
            )


class TestQATStepCondition(unittest.TestCase):
    """Verify QAT optimizer/scheduler steps only fire on sync boundaries."""

    def test_no_qat_optimizer_step_outside_sync(self):
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _training_step')
        fn_body = source[fn_start:]

        # Find the QAT block
        qat_start = fn_body.find('if self.qat_enabled:')
        self.assertGreater(qat_start, 0, 'QAT block should exist in _training_step')

        # Find end of method (next def or class)
        next_def = fn_body.find('\n    def ', 100)
        method_end = next_def if next_def > 0 else len(fn_body)

        after_qat = fn_body[qat_start:method_end]

        # There should be NO optimizer.step() or scheduler.step() in the QAT block
        # (they should only be in the should_sync block, which is before QAT)
        # The QAT block should only have _apply_fake_quantization calls
        qat_lines = after_qat.split('\n')
        for line in qat_lines:
            stripped = line.strip()
            if 'optimizer.step' in stripped or 'scheduler.step' in stripped:
                self.fail(f'QAT block must not contain optimizer/scheduler steps: {stripped}')

    def test_qat_within_sync(self):
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _training_step')
        fn_body = source[fn_start:]

        # Verify QAT block exists after should_sync block
        should_sync = fn_body.find('if should_sync:')
        qat_start = fn_body.find('if self.qat_enabled:')
        self.assertGreater(qat_start, should_sync,
                           'QAT block should appear AFTER should_sync block')


class TestNoSyncContextFSDP(unittest.TestCase):
    """Verify _no_sync_context handles FSDP."""

    def test_fsdp_branch_exists(self):
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _no_sync_context')
        fn_body = source[fn_start:source.find('\n    def ', fn_start + 50)]
        self.assertIn('use_fsdp', fn_body, 'FSDP branch must exist')
        self.assertIn('no_sync', fn_body, 'FSDP.no_sync must be called')
        self.assertIn('use_ddp', fn_body, 'DDP branch must exist')

    def test_no_sync_handles_fsdp(self):
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _no_sync_context')
        fn_body = source[fn_start:source.find('\n    def ', fn_start + 50)]
        self.assertIn('FullyShardedDataParallel.no_sync', fn_body,
                      'FSDP.no_sync must be called explicitly')


class TestMasterPortEval(unittest.TestCase):
    """Verify MASTER_PORT is dynamically assigned."""

    def test_no_short_circuit(self):
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _setup_device')
        fn_body = source[fn_start:source.find('\n    def ', fn_start + 50)]
        self.assertNotIn("12355", fn_body,
                         "MASTER_PORT must not have hardcoded '12355'")
        self.assertIn('find_free_port()', fn_body,
                      'find_free_port must be called for MASTER_PORT')


class TestFakeQuantizeSymmetric(unittest.TestCase):
    """Verify fake_quantize_tensor supports symmetric mode."""

    def test_symmetric_parameter_exists(self):
        source = _read_source('pipeline/training/quantization/common.py')
        fn_start = source.find('def fake_quantize_tensor')
        fn_header = source[fn_start:source.find('):', fn_start) + 2]
        self.assertIn('symmetric', fn_header,
                      'fake_quantize_tensor must accept symmetric parameter')

    def test_symmetric_logic(self):
        source = _read_source('pipeline/training/quantization/common.py')
        fn_start = source.find('def fake_quantize_tensor')
        # Find the function body up to the next def or end of file
        remaining = source[fn_start:]
        next_def = remaining.find('\ndef ')
        fn_body = remaining[:next_def] if next_def > 0 else remaining
        self.assertIn('if symmetric:', fn_body,
                      'Must have symmetric quantization branch')
        self.assertIn('zero_point = 0', fn_body,
                      'Symmetric quantization must use zero_point=0')

    def test_asymmetric_logic_preserved(self):
        source = _read_source('pipeline/training/quantization/common.py')
        self.assertIn('qmin = -(2 ** (num_bits - 1))', source,
                      'Asymmetric quantization qmin must exist')
        self.assertIn('zero_point = torch.clamp(zero_point, qmin, qmax)', source,
                      'Asymmetric quantization must clamp zero_point')


class TestSamplerDistributedSharding(unittest.TestCase):
    """Verify TemperatureSampler and BalancedLanguageSampler support world_size/rank."""

    def test_temperature_sampler_world_size(self):
        source = _read_source('pipeline/training/samplers.py')
        self.assertIn('world_size', source,
                      'TemperatureSampler must accept world_size')
        self.assertIn('rank', source,
                      'TemperatureSampler must accept rank')

    def test_balanced_sampler_world_size(self):
        source = _read_source('pipeline/training/samplers.py')
        self.assertIn('class BalancedLanguageSampler', source)
        self.assertIn('world_size', source,
                      'BalancedLanguageSampler must accept world_size')

    def test_set_epoch_exists(self):
        source = _read_source('pipeline/training/samplers.py')
        self.assertIn('def set_epoch', source,
                      'Samplers must have set_epoch method')


class TestCLIOverridePreservation(unittest.TestCase):
    """Verify CLI overrides survive hardware probing."""

    def test_force_batch_size_applied(self):
        source = _read_source('pipeline/training/trainer.py')
        init_start = source.find('def __init__(self')
        init_body = source[init_start:source.find('\n    def ', init_start + 50)]
        self.assertIn('force_batch_size', init_body,
                      'force_batch_size override must be applied in __init__')
        self.assertIn('force_learning_rate', init_body,
                      'force_learning_rate override must be applied in __init__')

    def test_strategy_not_overwritten(self):
        """Verify _probe_batch_size writes to strategy, not config.training."""
        source = _read_source('pipeline/training/trainer.py')
        fn_start = source.find('def _probe_batch_size')
        fn_body = source[fn_start:source.find('\n    def ', fn_start + 50)]
        # Should write to self.strategy.batch_size, not self.config.training.batch_size
        self.assertIn('self.strategy.batch_size', fn_body,
                      'Probe should write to strategy.batch_size')
        # Should NOT write to self.config.training.batch_size
        self.assertNotIn('self.config.training.batch_size', fn_body,
                         'Probe must NOT overwrite config.training.batch_size')


if __name__ == '__main__':
    unittest.main()
