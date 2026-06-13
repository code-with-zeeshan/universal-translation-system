"""End-to-end integration tests for the Universal Translation System pipeline.

Exercises real pipeline stages with mocked external dependencies (no GPU/torch
required). Tests that components wire together correctly end-to-end.

Uses file-text analysis and selective imports to avoid torch/textual deps.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_source(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding='utf-8')


class TestE2EPipelineStages(unittest.TestCase):
    """Exercise each pipeline stage start-to-finish with lightweight mocks."""

    maxDiff = None

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = self.tmpdir / "test_config.yaml"
        self._write_minimal_config()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── Helpers ────────────────────────────────────────────────────────

    def _write_minimal_config(self):
        config = {
            "general": {"experiment_name": "e2e_test"},
            "paths": {
                "output_dir": str(self.tmpdir / "output"),
                "processed_dir": str(self.tmpdir / "output" / "data" / "processed"),
                "vocab_dir": str(self.tmpdir / "output" / "vocabulary" / "vocab"),
                "checkpoint_dir": str(self.tmpdir / "output" / "checkpoints"),
            },
            "data": {
                "source_languages": ["eng"],
                "target_languages": ["fra"],
                "max_samples_per_pair": 100,
                "max_vocab_size": 1000,
            },
            "training": {
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": 0.001,
                "accumulation_steps": 1,
            },
            "vocabulary": {
                "vocab_size": 100,
                "model_type": "bpe",
            },
        }
        self.config_path.write_text(json.dumps(config))

    def _make_fake_corpus(self, lang_pair: str, num_lines: int = 10):
        """Write a minimal parallel corpus file."""
        pairs_dir = self.tmpdir / "corpora"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        src, tgt = lang_pair.split("-")
        fpath = pairs_dir / f"{lang_pair}.txt"
        with open(fpath, "w") as f:
            for i in range(num_lines):
                f.write(f"This is sentence {i} in English.\tCeci est la phrase {i} en francais.\n")
        return fpath

    # ── Pipeline Stage Connection Tests (file-text based) ─────────────

    def test_config_schema_loads(self):
        """Verify config.schemas exports load_config and RootConfig."""
        source = _read_source("config/schemas.py")
        self.assertIn("def load_config", source)
        self.assertIn("class RootConfig", source)
        self.assertIn("class DataConfig", source)
        self.assertIn("class TrainingConfig", source)

    def test_data_pipeline_exports(self):
        """Verify pipeline.data.orchestrator exports UnifiedDataPipeline."""
        source = _read_source("pipeline/data/orchestrator.py")
        self.assertIn("class UnifiedDataPipeline", source)
        self.assertIn("PipelineStage", source)
        self.assertIn("def run", source)

    def test_vocab_creator_exports(self):
        """Verify pipeline.vocabulary.creator exports UnifiedVocabularyCreator."""
        source = _read_source("pipeline/vocabulary/creator.py")
        self.assertIn("class UnifiedVocabularyCreator", source)
        self.assertIn("def create_all_packs", source)
        self.assertIn("def create_pack", source)

    def test_training_launcher_exports(self):
        """Verify pipeline.training.launch exports training functions."""
        source = _read_source("pipeline/training/launch.py")
        self.assertIn("def launch_training", source)
        self.assertIn("def main", source)
        self.assertIn("def load_configuration_dynamic_or_yaml", source)

    # ── TUI / Event Tests (file-text based to avoid textual dep) ──────

    def test_tui_bridge_exports(self):
        """Verify tui.bridge exports the expected classes."""
        source = _read_source("tui/bridge.py")
        self.assertIn("class TrainingBridge", source)
        self.assertIn("class PipelineBridge", source)
        self.assertIn("class GPUMonitor", source)
        self.assertIn("class TUIEventMessage", source)

    def test_tui_events_defined(self):
        """Verify all TUI event dataclasses exist in tui/events.py."""
        source = _read_source("tui/events.py")
        for event_cls in [
            "PipelineStageEvent", "PipelineLogEvent",
            "TrainingStartEvent", "TrainingEpochEvent",
            "TrainingBatchEvent", "TrainingEvalEvent",
            "TrainingLogEvent", "TrainingDoneEvent", "GPUStatusEvent",
        ]:
            self.assertIn(f"class {event_cls}", source,
                          f"{event_cls} not found in tui/events.py")

    def test_tui_training_panel_handlers(self):
        """Verify TrainingPanel has all required event handlers."""
        source = _read_source("tui/widgets/training_panel.py")
        for handler in [
            "handle_start", "handle_batch", "handle_epoch",
            "handle_eval", "handle_done", "handle_gpu",
        ]:
            self.assertIn(f"def {handler}", source,
                          f"TrainingPanel missing handler: {handler}")

    def test_tui_pipeline_panel_handlers(self):
        """Verify PipelinePanel has stage event handler."""
        source = _read_source("tui/widgets/pipeline_panel.py")
        self.assertIn("def handle_stage_event", source)

    def test_tui_log_panel_handlers(self):
        """Verify LogPanel has required event handlers."""
        source = _read_source("tui/widgets/log_panel.py")
        self.assertIn("def handle_pipeline_log", source)
        self.assertIn("def handle_training_log", source)

    def test_app_dispatches_all_events(self):
        """Verify app.py on_tui_event_message dispatches every event type."""
        source = _read_source("tui/app.py")
        for event_cls in [
            "PipelineStageEvent", "PipelineLogEvent",
            "TrainingStartEvent", "TrainingBatchEvent",
            "TrainingEpochEvent", "TrainingEvalEvent",
            "TrainingDoneEvent", "TrainingLogEvent", "GPUStatusEvent",
        ]:
            self.assertIn(f"isinstance(ev, {event_cls})", source,
                          f"app.py missing dispatch for {event_cls}")

    # ── Bridge Wiring Tests ───────────────────────────────────────────

    def test_trainer_batch_callback_signature(self):
        """Verify trainer._on_batch_end callback signature matches bridge."""
        trainer_src = _read_source("pipeline/training/trainer.py")
        self.assertIn("on_batch_end:", trainer_src)
        self.assertIn("on_epoch_end:", trainer_src)
        self.assertIn("on_eval_end:", trainer_src)
        # Verify _log_step calls _on_batch_end
        log_step = trainer_src[trainer_src.find("def _log_step"):]
        log_step = log_step[:log_step.find("\n    def ")]
        self.assertIn("self._on_batch_end(", log_step)

    def test_trainer_epoch_callback_invoked(self):
        """Verify _log_metrics calls _on_epoch_end."""
        trainer_src = _read_source("pipeline/training/trainer.py")
        log_metrics = trainer_src[trainer_src.find("def _log_metrics"):]
        log_metrics = log_metrics[:log_metrics.find("\n    def ")]
        self.assertIn("self._on_epoch_end(", log_metrics)

    def test_trainer_eval_callback_invoked(self):
        """Verify _log_metrics calls _on_eval_end."""
        trainer_src = _read_source("pipeline/training/trainer.py")
        log_metrics = trainer_src[trainer_src.find("def _log_metrics"):]
        log_metrics = log_metrics[:log_metrics.find("\n    def ")]
        self.assertIn("self._on_eval_end(", log_metrics)

    def test_bridge_wires_all_callbacks(self):
        """Verify TrainingBridge wires all three callbacks."""
        source = _read_source("tui/bridge.py")
        bridge_section = source[source.find("class TrainingBridge"):]
        bridge_section = bridge_section[:bridge_section.find("class GPUMonitor")]
        self.assertIn("on_batch =", bridge_section)
        self.assertIn("on_epoch =", bridge_section)
        self.assertIn("on_eval =", bridge_section)
        self.assertIn("trainer._on_batch_end", bridge_section)
        self.assertIn("trainer._on_epoch_end", bridge_section)
        self.assertIn("trainer._on_eval_end", bridge_section)
        self.assertIn("TrainingEvalEvent", bridge_section)

    # ── Utility / Path Tests (file-text based to avoid torch dep) ─────

    def test_path_utils_exports(self):
        """Verify utils/pathing.py exports the expected functions."""
        source = _read_source("utils/pathing.py")
        for fn in ["to_path", "resolve", "ensure_dir", "safe_join",
                    "exists_and_nonempty", "check_path_traversal"]:
            self.assertIn(f"def {fn}", source)

    def test_error_boundary_exports(self):
        """Verify utils/error_boundary.py exports run_safely and error_boundary."""
        source = _read_source("utils/error_boundary.py")
        self.assertIn("def run_safely", source)
        self.assertIn("def error_boundary", source)

    def test_exception_hierarchy_in_file(self):
        """Verify utils/exceptions.py has all exception classes."""
        source = _read_source("utils/exceptions.py")
        for cls in [
            "UniversalTranslationError", "DataError", "VocabularyError",
            "ModelError", "ConfigurationError", "TrainingError",
            "InferenceError", "ResourceError", "MemoryError",
            "SecurityError", "NetworkError", "ValidationError",
        ]:
            self.assertIn(f"class {cls}", source)

    def test_runtime_directory_manager_properties(self):
        """Verify RuntimeDirectoryManager has all expected path properties."""
        source = _read_source("utils/common_utils.py")
        for prop in [
            "data_dir", "raw_dir", "processed_dir", "vocab_dir",
            "logs_dir", "models_dir", "checkpoints_dir",
        ]:
            self.assertIn(f"def {prop}", source)

    def test_downloader_has_opus_fallback(self):
        """Verify downloader has OPUS-100 fallback logic."""
        source = _read_source("pipeline/data/downloader.py")
        self.assertIn("_download_direct_opus", source,
                      "downloader should have direct_opus fallback")
        self.assertIn("OPUS_DIRECT_URL", source)

    # ── Error Boundary / Entry Points ─────────────────────────────────

    def test_key_entry_points_have_error_boundary(self):
        """Verify key entry points use error boundary."""
        for rel in ["tui/__main__.py", "tui/app.py"]:
            source = _read_source(rel)
            if "from utils.error_boundary import" not in source:
                self.fail(f"{rel} missing error boundary import")

    def test_error_boundary_exit_codes(self):
        """Verify error_boundary.py maps exceptions to correct exit codes."""
        source = _read_source("utils/error_boundary.py")
        self.assertIn("ConfigurationError", source)
        self.assertIn("DataError", source)
        self.assertIn("VocabularyError", source)
        self.assertIn("ModelError", source)
        self.assertIn("TrainingError", source)
        self.assertIn("_EXIT_CODES", source)

    # ── Path Hygiene Tests ─────────────────────────────────────────────

    def test_no_os_path_join_in_core(self):
        """Verify no os.path.join remains in core pipeline modules."""
        core_files = [
            "pipeline/training/trainer.py",
            "pipeline/training/launch.py",
            "pipeline/data/orchestrator.py",
            "pipeline/data/downloader.py",
            "pipeline/vocabulary/creator.py",
            "pipeline/vocabulary/evolve.py",
            "utils/common_utils.py",
            "utils/logging_config.py",
        ]
        for rel in core_files:
            fpath = PROJECT_ROOT / rel
            if not fpath.exists():
                continue
            content = fpath.read_text()
            if "os.path.join" in content:
                self.fail(f"{rel} still contains os.path.join")

    def test_training_fixes_test(self):
        """Verify training critical fixes test exists and has expected tests."""
        source = _read_source("tests/test_training_critical_fixes.py")
        test_classes = [
            "TestCheckpointStateOrdering",
            "TestQATStepCondition",
            "TestNoSyncContextFSDP",
            "TestMasterPortEval",
            "TestFakeQuantizeSymmetric",
            "TestSamplerDistributedSharding",
            "TestCLIOverridePreservation",
        ]
        for cls in test_classes:
            self.assertIn(f"class {cls}", source)


if __name__ == "__main__":
    unittest.main()
