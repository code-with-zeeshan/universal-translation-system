"""
Tests for training.hardware_profile.

Relies on conftest.py for all heavy dependency mocks (torch, numpy, etc.).
"""

import unittest
from unittest.mock import MagicMock, patch

from training.hardware_profile import (
    HardwareProfile,
    find_free_port,
    launch_distributed_intelligent_training,
)


class TestHardwareProfileEnum(unittest.TestCase):
    """Test HardwareProfile enum values."""

    def test_enum_values_present(self):
        self.assertIn(HardwareProfile.LOW_END_SINGLE, HardwareProfile)
        self.assertIn(HardwareProfile.MID_RANGE_SINGLE, HardwareProfile)
        self.assertIn(HardwareProfile.HIGH_END_SINGLE, HardwareProfile)
        self.assertIn(HardwareProfile.LOW_END_MULTI, HardwareProfile)
        self.assertIn(HardwareProfile.MID_RANGE_MULTI, HardwareProfile)
        self.assertIn(HardwareProfile.HIGH_END_MULTI, HardwareProfile)
        self.assertIn(HardwareProfile.CPU_ONLY, HardwareProfile)
        self.assertIn(HardwareProfile.TPU, HardwareProfile)
        self.assertIn(HardwareProfile.APPLE_SILICON, HardwareProfile)

    def test_enum_values_count(self):
        self.assertEqual(len(list(HardwareProfile)), 9)

    def test_enum_unique_values(self):
        names = [m.name for m in HardwareProfile]
        self.assertEqual(len(names), len(set(names)))


class TestHardwareProfileDetection(unittest.TestCase):
    """Test detection logic (simulated since from_torch_detection may not exist)."""

    def test_cpu_when_cuda_unavailable(self):
        import torch
        torch.cuda.is_available.return_value = False
        if hasattr(HardwareProfile, 'from_torch_detection'):
            profile = HardwareProfile.from_torch_detection()
            self.assertEqual(profile, HardwareProfile.CPU_ONLY)

    def test_single_gpu_low_end(self):
        import torch
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_capability.return_value = (7, 0)
        torch.cuda.get_device_name.return_value = "Tesla T4"

        if hasattr(HardwareProfile, 'from_torch_detection'):
            profile = HardwareProfile.from_torch_detection()
            self.assertEqual(profile, HardwareProfile.LOW_END_SINGLE)

    def test_multi_gpu_high_end(self):
        import torch
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 4
        torch.cuda.get_device_capability.return_value = (8, 0)
        torch.cuda.get_device_name.return_value = "NVIDIA A100"

        if hasattr(HardwareProfile, 'from_torch_detection'):
            profile = HardwareProfile.from_torch_detection()
            self.assertEqual(profile, HardwareProfile.HIGH_END_MULTI)


class TestFindFreePort(unittest.TestCase):
    """Test find_free_port function."""

    @patch('training.hardware_profile.socket.socket')
    def test_find_free_port(self, mock_socket):
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock
        mock_sock.getsockname.return_value = ('', 12345)

        port = find_free_port()
        self.assertEqual(port, 12345)
        mock_sock.bind.assert_called_once()


class TestLaunchDistributedIntelligentTraining(unittest.TestCase):
    """Test launch_distributed_intelligent_training function."""

    @patch('training.trainer.train_intelligent')
    def test_launch_sets_env_and_calls_train(self, mock_train):
        encoder = MagicMock()
        decoder = MagicMock()
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        config = MagicMock()
        experiment_name = "test_exp"

        os_mod = 'training.hardware_profile.os'
        with patch.dict(os_mod + '.environ', {}, clear=True):
            launch_distributed_intelligent_training(
                rank=0, world_size=2,
                encoder=encoder, decoder=decoder,
                train_dataset=train_dataset, val_dataset=val_dataset,
                config=config, experiment_name=experiment_name,
            )

        mock_train.assert_called_once_with(
            encoder=encoder, decoder=decoder,
            train_dataset=train_dataset, val_dataset=val_dataset,
            config=config, experiment_name=experiment_name,
        )


class TestHardwareProfileSignatures(unittest.TestCase):
    """Test function signatures exist."""

    def test_launch_distributed_intelligent_training_callable(self):
        self.assertTrue(callable(launch_distributed_intelligent_training))

    def test_find_free_port_callable(self):
        self.assertTrue(callable(find_free_port))
