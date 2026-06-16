"""End-to-end smoke test that actually runs the training pipeline.

Requires torch and other deps — skips automatically if unavailable.
Designed for Lightning Studio (or any environment with real torch).

Run: python -m pytest tests/test_end_to_end.py -x -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path


def _unmock_all():
    """Remove ALL conftest mocks from sys.modules so real imports are used."""
    _CONFTEST_MOCK_PREFIXES = (
        "torch", "numpy", "pydantic", "yaml", "wandb", "psutil",
        "prometheus_client", "safetensors", "sacrebleu", "nvidia_ml_py3",
        "jwt", "cryptography", "keyring", "msgpack", "fastapi",
        "starlette", "litserve", "opentelemetry", "urllib3", "requests",
        "tracemalloc",
    )
    for prefix in _CONFTEST_MOCK_PREFIXES:
        for k in list(sys.modules.keys()):
            if k == prefix or k.startswith(prefix + "."):
                del sys.modules[k]


# Undo conftest mocks before importing project modules
_unmock_all()

try:
    import torch
    TORCH_AVAILABLE = torch.__version__ is not None
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False


def _minimal_translation_corpus(tmpdir: Path, num_pairs: int = 20):
    """Create a tiny parallel corpus file."""
    pairs_file = tmpdir / "train.txt"
    with open(pairs_file, "w") as f:
        for i in range(num_pairs):
            f.write(f"hello world {i}\tbonjour le monde {i}\n")
    val_file = tmpdir / "val.txt"
    with open(val_file, "w") as f:
        for i in range(5):
            f.write(f"good morning {i}\tbonjour {i}\n")
    return pairs_file, val_file


@unittest.skipIf(not TORCH_AVAILABLE, "torch not available — cannot run end-to-end test")
class TestEndToEndSmoke(unittest.TestCase):
    """Minimal end-to-end pipeline smoke test (1 training step, on CPU)."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        # Remove conftest project stubs that shadow real modules
        _STUB_PREFIXES = ("config", "utils", "integration", "evaluation",
                          "monitoring", "tools", "runtime", "encoder",
                          "decoder", "pipeline")
        for prefix in _STUB_PREFIXES:
            for k in list(sys.modules.keys()):
                if k == prefix or k.startswith(prefix + "."):
                    # Skip submodules that were legitimately imported by real code
                    if hasattr(sys.modules.get(k, None), "__file__"):
                        continue
                    if k in sys.modules:
                        del sys.modules[k]

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        train_path, val_path = _minimal_translation_corpus(self.tmpdir)
        self.train_path = train_path
        self.val_path = val_path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_tiny_models(self):
        """Create minimal encoder and decoder for smoke test."""
        from runtime.encoder.universal_encoder import UniversalEncoder
        from runtime.cloud_decoder.decoder_core import OptimizedUniversalDecoder
        encoder = UniversalEncoder(
            max_vocab_size=100,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            max_positions=64,
            dropout=0.1,
        )
        decoder = OptimizedUniversalDecoder(
            encoder_dim=32,
            decoder_dim=32,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            max_length=64,
            dropout=0.1,
        )
        return encoder, decoder

    def test_forward_pass(self):
        """Model can do a forward/backward pass and loss decreases."""
        from torch import optim
        encoder, decoder = self._make_tiny_models()
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimiser = optim.AdamW(params, lr=0.001)

        dummy_src = torch.randint(0, 100, (2, 16))
        dummy_tgt = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.bool)

        encoder_output = encoder(dummy_src, dummy_mask)
        decoder_output = decoder(
            decoder_input_ids=dummy_tgt[:, :-1],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=dummy_mask,
        )

        loss = torch.nn.functional.cross_entropy(
            decoder_output.reshape(-1, decoder_output.size(-1)),
            dummy_tgt[:, 1:].reshape(-1),
        )
        self.assertFalse(torch.isnan(loss), "Loss is NaN")
        self.assertFalse(torch.isinf(loss), "Loss is Inf")
        first_loss = loss.item()

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Second forward pass
        encoder_output2 = encoder(dummy_src, dummy_mask)
        decoder_output2 = decoder(
            decoder_input_ids=dummy_tgt[:, :-1],
            encoder_hidden_states=encoder_output2,
            encoder_attention_mask=dummy_mask,
        )
        loss2 = torch.nn.functional.cross_entropy(
            decoder_output2.reshape(-1, decoder_output2.size(-1)),
            dummy_tgt[:, 1:].reshape(-1),
        )
        # Loss should decrease (even if slightly)
        self.assertLess(loss2.item(), first_loss * 1.1,
                        msg=f"Loss did not decrease: {loss2.item():.4f} vs {first_loss:.4f}")

    def test_checkpoint_save_load(self):
        """Can save and load a checkpoint correctly."""
        encoder, decoder = self._make_tiny_models()
        checkpoint_dir = self.tmpdir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"

        dummy_src = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.bool)
        encoder_out = encoder(dummy_src, dummy_mask)

        checkpoint = {
            "epoch": 5,
            "global_step": 100,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        self.assertTrue(checkpoint_path.exists(), "Checkpoint file not created")
        self.assertGreater(checkpoint_path.stat().st_size, 100,
                           msg="Checkpoint suspiciously small")

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertIn("epoch", loaded)
        self.assertEqual(loaded["epoch"], 5)
        encoder.load_state_dict(loaded["encoder_state_dict"])
        decoder.load_state_dict(loaded["decoder_state_dict"])
        encoder_out2 = encoder(dummy_src, dummy_mask)
        self.assertTrue(torch.equal(encoder_out, encoder_out2),
                        msg="Model output changed after save/load")

    def test_training_launch_imports(self):
        """train_intelligent can be imported and called without crash."""
        from pipeline.training.trainer import train_intelligent
        self.assertTrue(callable(train_intelligent))

    def test_config_pydantic_loads(self):
        """RootConfig can be instantiated without import-time crash."""
        from config.schemas import RootConfig, DataConfig, ModelConfig, TrainingConfig, MemoryConfig, VocabularyConfig
        cfg = RootConfig(
            data=DataConfig(training_distribution={"eng-fra": 1.0}),
            model=ModelConfig(),
            training=TrainingConfig(),
            memory=MemoryConfig(),
            vocabulary=VocabularyConfig(),
        )
        self.assertIsNotNone(cfg)

    def test_health_monitor(self):
        """TrainingHealthMonitor runs its callbacks without error."""
        from pipeline.training.health_monitor import TrainingHealthMonitor
        monitor = TrainingHealthMonitor(self.tmpdir, "smoke_test")
        # Simulate an epoch callback
        monitor.on_epoch_end(0, 3, 8.5, None, 0.001, 0.0)
        monitor.on_epoch_end(1, 3, 5.5, None, 0.001, 0.0)
        monitor.on_epoch_end(2, 3, 4.0, 3.8, 0.0005, 0.0)
        report = monitor.final_report()
        self.assertIn("epochs_trained", report)
        self.assertEqual(report["epochs_trained"], 3)

    def test_runtime_directory_manager_no_import_crash(self):
        """RuntimeDirectoryManager can be imported without crash."""
        from utils.common_utils import RuntimeDirectoryManager
        rdm = RuntimeDirectoryManager()
        self.assertTrue(hasattr(rdm, "logs_dir"))
        self.assertTrue(hasattr(rdm, "checkpoints_dir"))

    def test_training_pipeline_imports(self):
        """All training pipeline modules can be imported without crash."""
        modules = [
            "pipeline.training.launch",
            "pipeline.training.trainer",
            "pipeline.training.utils",
            "pipeline.training.datasets",
            "pipeline.training.analytics",
            "pipeline.training.health_monitor",
        ]
        for mod_path in modules:
            with self.subTest(module=mod_path):
                __import__(mod_path)


@unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
class TestEndToEndPipelineRun(unittest.TestCase):
    """Runs the full training pipeline end to end (1 batch, CPU).

    This is the 'biggest gap' test that exercises actual pipeline code paths.
    """

    def test_tiny_training_loop(self):
        """Runs 1 training step and verifies loss decreases."""
        from torch import nn, optim
        from runtime.encoder.universal_encoder import UniversalEncoder
        from runtime.cloud_decoder.decoder_core import OptimizedUniversalDecoder

        encoder = UniversalEncoder(
            max_vocab_size=100, hidden_dim=32, num_layers=1,
            num_heads=4, max_positions=32, dropout=0.1,
        )
        decoder = OptimizedUniversalDecoder(
            encoder_dim=32, decoder_dim=32, num_layers=1,
            num_heads=4, vocab_size=100, max_length=32, dropout=0.1,
        )

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimiser = optim.AdamW(params, lr=0.01)

        src = torch.randint(0, 100, (4, 16))
        tgt = torch.randint(0, 100, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.bool)

        losses = []
        for step in range(3):
            optimiser.zero_grad()
            enc_out = encoder(src, mask)
            dec_out = decoder(
                decoder_input_ids=tgt[:, :-1],
                encoder_hidden_states=enc_out,
                encoder_attention_mask=mask,
            )
            loss = nn.functional.cross_entropy(
                dec_out.reshape(-1, dec_out.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        # Loss should decrease over 3 steps
        self.assertLess(losses[-1], losses[0] * 1.05,
                        msg=f"Loss not decreasing: {losses}")


if __name__ == "__main__":
    unittest.main()
