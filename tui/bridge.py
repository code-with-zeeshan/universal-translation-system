# tui/bridge.py
"""Bridge between pipeline/trainer and TUI event system.

Wraps existing components without modifying them.
Usage: register an `on_event` callback, then call `run_pipeline()` or
       `run_training()` — they return as soon as the operation completes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tui.events import (
    GPUStatusEvent,
    PipelineLogEvent,
    PipelineStageEvent,
    StageStatus,
    TrainingBatchEvent,
    TrainingDoneEvent,
    TrainingEpochEvent,
    TrainingLogEvent,
    TrainingStartEvent,
    TUIEvent,
)

OnEvent = Callable[[TUIEvent], None]

# ── Log interceptor ───────────────────────────────────────────────────────

_LOG_LEVEL_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class _TUILogHandler(logging.Handler):
    """Redirects log records to the TUI event stream."""

    def __init__(self, on_event: OnEvent) -> None:
        super().__init__(level=logging.DEBUG)
        self._on_event = on_event

    def emit(self, record: logging.Record) -> None:
        try:
            self._on_event(PipelineLogEvent(
                level=_LOG_LEVEL_MAP.get(record.levelno, "INFO"),
                message=self.format(record),
                logger=record.name,
            ))
        except Exception:
            self.handleError(record)


class _TrainingLogHandler(logging.Handler):
    """Redirects training log records to the TUI event stream as raw text."""

    def __init__(self, on_event: OnEvent) -> None:
        super().__init__(level=logging.DEBUG)
        self._on_event = on_event

    def emit(self, record: logging.Record) -> None:
        try:
            self._on_event(TrainingLogEvent(
                level=_LOG_LEVEL_MAP.get(record.levelno, "INFO"),
                message=self.format(record),
            ))
        except Exception:
            self.handleError(record)


# ── Pipeline Bridge ───────────────────────────────────────────────────────

class PipelineBridge:
    """Wraps UnifiedDataPipeline, emitting stage/log events to the TUI."""

    def __init__(self, on_event: OnEvent, config_path: str) -> None:
        self._on_event = on_event
        self._config_path = config_path
        self._log_handler: Optional[_TUILogHandler] = None

    async def run(self, stages: Optional[list[str]] = None) -> None:
        """Run the pipeline with TUI event emission."""
        from pipeline.data.orchestrator import UnifiedDataPipeline, PipelineStage

        # Attach log interceptor
        root_logger = logging.getLogger()
        self._log_handler = _TUILogHandler(self._on_event)
        self._log_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(self._log_handler)

        try:
            # Initialize pipeline
            pipeline = UnifiedDataPipeline(config_path=self._config_path)

            # Determine stages
            stage_list: list[PipelineStage] = []
            if stages:
                for s in stages:
                    try:
                        stage_list.append(PipelineStage(s))
                    except ValueError:
                        pass
            else:
                self._on_event(PipelineStageEvent(
                    stage="pipeline", status=StageStatus.RUNNING,
                    message="Running full pipeline",
                ))
                await pipeline.run()
                self._on_event(PipelineStageEvent(
                    stage="pipeline", status=StageStatus.DONE,
                    progress=1.0, message="Pipeline complete",
                ))
                return

            # Run selected stages sequentially with event emission
            for i, stage in enumerate(stage_list):
                stage_name = stage.value
                progress_base = i / len(stage_list)
                self._on_event(PipelineStageEvent(
                    stage=stage_name, status=StageStatus.RUNNING,
                    progress=progress_base,
                ))
                try:
                    await pipeline.run_single_stage(stage)
                    self._on_event(PipelineStageEvent(
                        stage=stage_name, status=StageStatus.DONE,
                        progress=(i + 1) / len(stage_list),
                        message=f"{stage_name} completed",
                    ))
                except Exception as exc:
                    self._on_event(PipelineStageEvent(
                        stage=stage_name, status=StageStatus.FAILED,
                        progress=progress_base,
                        message=str(exc),
                    ))
                    raise

        finally:
            if self._log_handler:
                root_logger.removeHandler(self._log_handler)


# ── Training Bridge ───────────────────────────────────────────────────────

class TrainingBridge:
    """Wraps IntelligentTrainer, emitting training events to the TUI."""

    def __init__(self, on_event: OnEvent, config_path: str) -> None:
        self._on_event = on_event
        self._config_path = config_path
        self._log_handler: Optional[_TrainingLogHandler] = None

    async def run(self) -> dict[str, Any]:
        """Run training with TUI event emission."""
        from pipeline.training.trainer import IntelligentTrainer
        from config.schemas import load_config

        # Load config
        config = load_config(self._config_path)

        # Attach log interceptor for raw text log forwarding
        root_logger = logging.getLogger()
        self._log_handler = _TrainingLogHandler(self._on_event)
        self._log_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(self._log_handler)

        start_time = time.time()

        # Build structured event callbacks
        on_batch = (lambda ep, bi, tb, l, lr, tps:
            self._on_event(TrainingBatchEvent(
                epoch=ep, batch=bi, total_batches=tb,
                loss=l, learning_rate=lr, tokens_per_second=tps,
            )))
        on_epoch = (lambda ep, te, trl, vl, lr, tps:
            self._on_event(TrainingEpochEvent(
                epoch=ep, total_epochs=te,
                train_loss=trl, val_loss=vl,
                learning_rate=lr, tokens_per_second=tps,
            )))
        on_eval = (lambda ep, vl, trl, bleu, chrf, comet:
            self._on_event(TrainingEvalEvent(
                epoch=ep, val_loss=vl, bleu=bleu or 0.0,
                chrf=chrf, comet_score=comet,
            )))

        try:
            # Emit start event
            self._on_event(TrainingStartEvent(
                total_epochs=config.training.num_epochs,
                batch_size=config.training.batch_size,
                effective_batch_size=config.training.batch_size * config.training.accumulation_steps,
                learning_rate=config.training.lr,
            ))

            # Initialize and run trainer with structured callbacks
            trainer = IntelligentTrainer(config)
            trainer._on_batch_end = on_batch
            trainer._on_epoch_end = on_epoch
            trainer._on_eval_end = on_eval
            result = trainer.train()

            total_time = time.time() - start_time
            best_bleu = result.get("best_bleu", 0.0)
            best_epoch = result.get("best_epoch", 0)

            self._on_event(TrainingDoneEvent(
                best_bleu=best_bleu,
                best_epoch=best_epoch,
                total_time_s=total_time,
                model_path=result.get("model_path", ""),
            ))

            return result

        finally:
            if self._log_handler:
                root_logger.removeHandler(self._log_handler)


# ── GPU Monitor ───────────────────────────────────────────────────────────

class GPUMonitor:
    """Periodically snapshots GPU status and emits GPUStatusEvent."""

    def __init__(self, on_event: OnEvent, interval_s: float = 2.0) -> None:
        self._on_event = on_event
        self._interval = interval_s
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                self._on_event(GPUStatusEvent(
                    gpu_util=float(util.gpu),
                    memory_used_gb=mem_info.used / (1024**3),
                    memory_total_gb=mem_info.total / (1024**3),
                    temperature_c=float(temp),
                ))
            except Exception:
                pass
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False


# ── Textual Message (for posting events into Textual's message system) ─────

from textual.message import Message

class TUIEventMessage(Message):
    """Wraps a TUIEvent for Textual's message pump."""
    def __init__(self, event: TUIEvent) -> None:
        super().__init__()
        self.tui_event = event
