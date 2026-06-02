# tui/app.py
"""Universal Translation System — Terminal UI Dashboard.

Run with:
    python -m tui.app --config config/base.yaml              # full pipeline then train
    python -m tui.app --config config/base.yaml --pipeline    # pipeline only
    python -m tui.app --config config/base.yaml --train       # training only
"""

from __future__ import annotations

import asyncio
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState, get_current_worker

from tui.bridge import (
    GPUMonitor,
    PipelineBridge,
    TrainingBridge,
    TUIEventMessage,
)
from tui.events import (
    GPUStatusEvent,
    PipelineLogEvent,
    PipelineStageEvent,
    StageStatus,
    TrainingBatchEvent,
    TrainingDoneEvent,
    TrainingEpochEvent,
    TrainingEvalEvent,
    TrainingLogEvent,
    TrainingStartEvent,
    TUIEvent,
)
from tui.widgets.log_panel import LogPanel
from tui.widgets.pipeline_panel import PipelinePanel
from tui.widgets.training_panel import TrainingPanel


# ── CSS ───────────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    layout: grid;
    grid-size: 2 2;
    grid-rows: 1fr 1fr;
    grid-columns: 1fr 1fr;
}

#pipeline-panel {
    border: solid $primary;
    padding: 0 1;
    overflow: auto;
}

#training-panel {
    border: solid $secondary;
    padding: 0 1;
    overflow: auto;
}

#log-panel {
    column-span: 2;
    border: solid $surface;
    padding: 0 1;
}

.panel-header {
    text-style: bold;
    margin: 0 0 1 0;
    color: $text-muted;
}

.stage-row {
    margin: 0 0 0 0;
}

.pipeline-summary {
    margin: 1 0 0 0;
    text-style: italic;
    color: $text-disabled;
}

.training-history {
    margin: 1 0 0 0;
    color: $accent;
}

#log-view {
    height: 1fr;
}
"""


# ── Main Screen ───────────────────────────────────────────────────────────

class MainScreen(Screen[None]):
    """Dashboard with pipeline, training, and log panels."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("h", "help", "Help"),
    ]

    def __init__(self, config_path: str, run_mode: str) -> None:
        super().__init__()
        self._config_path = config_path
        self._run_mode = run_mode  # "all" | "pipeline" | "train"
        self._pipeline_panel: Optional[PipelinePanel] = None
        self._training_panel: Optional[TrainingPanel] = None
        self._log_panel: Optional[LogPanel] = None
        self._gpu_monitor: Optional[GPUMonitor] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self._pipeline_panel = PipelinePanel()
        yield self._pipeline_panel
        self._training_panel = TrainingPanel()
        yield self._training_panel
        self._log_panel = LogPanel()
        yield self._log_panel
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Universal Translation System"
        self.sub_title = f"config: {Path(self._config_path).name}"

        # Start GPU monitor
        self._gpu_monitor = GPUMonitor(self._post_event, interval_s=2.0)
        asyncio.ensure_future(self._gpu_monitor.start())

        # Start the pipeline/training in background workers
        if self._run_mode in ("all", "pipeline"):
            self.run_pipeline_worker()
        elif self._run_mode == "train":
            self.run_training_worker()

    def on_unmount(self) -> None:
        if self._gpu_monitor:
            self._gpu_monitor.stop()

    # ── Event dispatch (called from bridge threads) ──────────────────────

    def _post_event(self, event: TUIEvent) -> None:
        """Called by bridge/GPU monitor — posts to Textual message queue."""
        self.post_message(TUIEventMessage(event))

    def on_tui_event_message(self, msg: TUIEventMessage) -> None:
        """Handle incoming TUI events from the message queue."""
        ev = msg.tui_event
        # Dispatch by type
        if isinstance(ev, PipelineStageEvent):
            if self._pipeline_panel:
                self._pipeline_panel.handle_stage_event(ev)
        elif isinstance(ev, PipelineLogEvent):
            if self._log_panel:
                self._log_panel.handle_pipeline_log(ev)
        elif isinstance(ev, TrainingStartEvent):
            if self._training_panel:
                self._training_panel.handle_start(ev)
        elif isinstance(ev, TrainingBatchEvent):
            if self._training_panel:
                self._training_panel.handle_batch(ev)
        elif isinstance(ev, TrainingEpochEvent):
            if self._training_panel:
                self._training_panel.handle_epoch(ev)
        elif isinstance(ev, TrainingEvalEvent):
            if self._training_panel:
                self._training_panel.handle_eval(ev)
        elif isinstance(ev, TrainingDoneEvent):
            if self._training_panel:
                self._training_panel.handle_done(ev)
        elif isinstance(ev, TrainingLogEvent):
            if self._log_panel:
                self._log_panel.handle_training_log(ev)
        elif isinstance(ev, GPUStatusEvent):
            if self._training_panel:
                self._training_panel.handle_gpu(ev)

    # ── Background workers ───────────────────────────────────────────────

    @work(thread=False, group="pipeline", exit_on_error=False)
    async def run_pipeline_worker(self) -> None:
        """Run the pipeline in a background asyncio task."""
        self._post_event(PipelineStageEvent(
            stage="pipeline", status=StageStatus.RUNNING,
            message="Starting data pipeline...",
        ))
        bridge = PipelineBridge(self._post_event, self._config_path)
        try:
            await bridge.run()
            self._post_event(PipelineStageEvent(
                stage="pipeline",                 status=StageStatus.DONE,
                progress=1.0, message="Pipeline complete",
            ))
            # If mode is "all", start training after pipeline finishes
            if self._run_mode == "all":
                self.run_training_worker()
        except Exception as exc:
            self._post_event(PipelineStageEvent(
                stage="pipeline",                 status=StageStatus.FAILED,
                message=str(exc),
            ))

    @work(thread=False, group="training", exit_on_error=False)
    async def run_training_worker(self) -> None:
        """Run training in a background asyncio task."""
        bridge = TrainingBridge(self._post_event, self._config_path)
        try:
            await bridge.run()
        except Exception as exc:
            self._post_event(TrainingLogEvent(
                level="ERROR", message=f"Training failed: {exc}",
            ))

    # ── Actions ──────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        """Force-refresh all panels."""
        if self._pipeline_panel:
            self._pipeline_panel.refresh()
        if self._training_panel:
            self._training_panel.refresh()
        if self._log_panel:
            self._log_panel.refresh()

    def action_help(self) -> None:
        """Show help overlay."""
        from textual.screen import ModalScreen
        from textual.widgets import Label

        class HelpScreen(ModalScreen[None]):
            def compose(self):
                yield Label(
                    "  Keyboard Shortcuts\n"
                    "  ──────────────────\n"
                    "  q     — Quit\n"
                    "  r     — Refresh display\n"
                    "  h     — This help\n"
                    "  Esc   — Close help\n"
                    "\n"
                    "  Run Modes:\n"
                    "  ──────────\n"
                    "  --pipeline   Data pipeline only\n"
                    "  --train      Training only\n"
                    "  (default)    Pipeline then training\n"
                    "\n"
                    "  Config:  --config path/to/config.yaml\n"
                )
            def key_escape(self): self.dismiss()

        self.push_screen(HelpScreen())


# ── Entry Point ───────────────────────────────────────────────────────────

class TUIApp(App[None]):
    """Root application."""

    CSS = APP_CSS
    SCREENS = {"main": MainScreen}

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config_path: str, run_mode: str) -> None:
        super().__init__()
        self._config_path = config_path
        self._run_mode = run_mode

    def on_mount(self) -> None:
        self.push_screen("main")


def parse_args(argv: list[str] | None = None) -> Namespace:
    p = ArgumentParser(description="Universal Translation System TUI")
    p.add_argument("--config", default="config/base.yaml", help="Config file path")
    p.add_argument("--pipeline", action="store_true", help="Data pipeline only")
    p.add_argument("--train", action="store_true", help="Training only")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.pipeline and args.train:
        print("Use --pipeline OR --train, not both (default is both)")
        sys.exit(1)
    mode = "pipeline" if args.pipeline else ("train" if args.train else "all")
    app = TUIApp(config_path=args.config, run_mode=mode)
    app.run()


if __name__ == "__main__":
    main()
