# tui/events.py
"""Event dataclasses for pipeline and training progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


# ── Status Enum ──────────────────────────────────────────────────────────

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrainingPhase(Enum):
    INIT = "initializing"
    TRAINING = "training"
    EVAL = "evaluating"
    SAVING = "saving"
    DONE = "done"


# ── Pipeline Events ──────────────────────────────────────────────────────

@dataclass
class PipelineStageEvent:
    """Emitted when a pipeline stage changes status."""
    stage: str
    status: StageStatus
    progress: float = 0.0          # 0.0 – 1.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineLogEvent:
    """A log line from the pipeline."""
    level: str                      # INFO / WARNING / ERROR / DEBUG
    message: str
    logger: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ── Training Events ──────────────────────────────────────────────────────

@dataclass
class TrainingStartEvent:
    """Training has started."""
    total_epochs: int
    batch_size: int
    effective_batch_size: int
    learning_rate: float
    model_size_params: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingEpochEvent:
    """Emitted at the end of each training epoch."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingBatchEvent:
    """Emitted periodically during training (every N batches)."""
    epoch: int
    batch: int
    total_batches: int
    loss: float
    learning_rate: float
    tokens_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingEvalEvent:
    """Emitted after evaluation on validation set."""
    epoch: int
    bleu: float
    val_loss: float
    chrf: Optional[float] = None
    comet_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingLogEvent:
    """A log line from the trainer."""
    level: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingDoneEvent:
    """Training finished."""
    best_bleu: float
    best_epoch: int
    total_time_s: float
    model_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ── GPU / Resource Events ────────────────────────────────────────────────

@dataclass
class GPUStatusEvent:
    """GPU utilization and memory snapshot."""
    gpu_util: float = 0.0          # 0–100 %
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature_c: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# ── Union type for dispatch ──────────────────────────────────────────────

TUIEvent = (
    PipelineStageEvent
    | PipelineLogEvent
    | TrainingStartEvent
    | TrainingEpochEvent
    | TrainingBatchEvent
    | TrainingEvalEvent
    | TrainingLogEvent
    | TrainingDoneEvent
    | GPUStatusEvent
)
