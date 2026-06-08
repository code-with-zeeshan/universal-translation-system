"""Shared pipeline checkpoint utilities for data → train → eval auto-resume."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def hash_config(section: object) -> str:
    """SHA-256 fingerprint of a config section."""
    raw = json.dumps(section, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def checkpoint_path(phase: str, base_dir: str = ".") -> Path:
    """Path to checkpoint file for a pipeline phase."""
    return Path(base_dir) / f"{phase}_checkpoint.json"


class PhaseCheckpoint:
    """Unified checkpoint for a pipeline phase (data/train/eval)."""

    def __init__(self, phase: str, base_dir: str = "."):
        self.phase = phase
        self.path = checkpoint_path(phase, base_dir)
        self.data: dict = {}

    def load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
                logger.info(f"📋 Loaded {self.phase} checkpoint ({self._summarize()})")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {self.phase} checkpoint: {e}")
                self.data = {}
        return self.data

    def save(self, **updates):
        self.data.update(updates)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        logger.debug(f"💾 Saved {self.phase} checkpoint")

    @property
    def completed(self) -> bool:
        return self.data.get("completed", False)

    @completed.setter
    def completed(self, value: bool):
        self.data["completed"] = value

    @property
    def config_hash(self) -> str:
        return self.data.get("config_hash", "")

    @config_hash.setter
    def config_hash(self, value: str):
        self.data["config_hash"] = value

    def is_up_to_date(self, current_hash: str) -> bool:
        """True if checkpoint exists, completed, and config unchanged."""
        return self.completed and self.config_hash == current_hash

    def reset(self):
        self.data = {}
        self.save()
        logger.info(f"🔄 Reset {self.phase} checkpoint")

    def _summarize(self) -> str:
        c = self.data.get("completed", False)
        h = self.config_hash[:8] if self.config_hash else ""
        return f"completed={c}, hash={h}"


# ── Cross-stage pipeline tracking ─────────────────────────────

PIPELINE_STATE_PATH = Path("pipeline_state.json")

STAGES = ["data", "train", "eval"]
STAGE_DEPENDENCIES = {
    "train": ["data"],
    "eval": ["train"],
}


def load_pipeline_state() -> dict:
    """Load cross-stage pipeline state (root-level pipeline_state.json)."""
    if PIPELINE_STATE_PATH.exists():
        try:
            with open(PIPELINE_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_pipeline_state(state: dict):
    """Save cross-stage pipeline state."""
    with open(PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def mark_stage_complete(stage: str, config_hash: str, extra: Optional[Dict] = None):
    """Mark a pipeline stage as fully complete in the global tracker."""
    state = load_pipeline_state()
    entry = {"completed": True, "config_hash": config_hash}
    if extra:
        entry.update(extra)
    state[stage] = entry
    save_pipeline_state(state)
    logger.info(f"✅ Pipeline stage '{stage}' marked complete")


def is_stage_complete(stage: str, config_hash: str) -> bool:
    """Check if a stage is complete with matching config hash."""
    state = load_pipeline_state()
    entry = state.get(stage, {})
    return entry.get("completed", False) and entry.get("config_hash") == config_hash


def invalidate_downstream(from_stage: str):
    """When a stage is re-run, mark all downstream stages as incomplete."""
    state = load_pipeline_state()
    found = False
    for s in STAGES:
        if s == from_stage:
            found = True
        if found:
            if s in state:
                state[s]["completed"] = False
                logger.info(f"⏳ Invalidated downstream stage '{s}'")
    save_pipeline_state(state)


def reset_pipeline_state():
    """Clear the global pipeline state entirely."""
    if PIPELINE_STATE_PATH.exists():
        PIPELINE_STATE_PATH.unlink()
    logger.info("🔄 Global pipeline state reset")
