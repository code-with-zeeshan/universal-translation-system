# tui/widgets/log_panel.py
"""Scrolling log panel at the bottom."""

from __future__ import annotations

from textual.widgets import RichLog
from textual.widget import Widget
from rich.text import Text
from rich.style import Style

from tui.events import PipelineLogEvent, TrainingLogEvent


# ── Level styles ──────────────────────────────────────────────────────────

_LEVEL_STYLE = {
    "DEBUG": "dim",
    "INFO": "",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}


# ── Log Panel ─────────────────────────────────────────────────────────────

class LogPanel(Widget):
    """Scrollable log at the bottom of the screen."""

    MAX_LINES = 500

    def __init__(self) -> None:
        super().__init__(id="log-panel")
        self._log = RichLog(
            id="log-view",
            highlight=True,
            markup=False,
            max_lines=self.MAX_LINES,
        )

    def compose(self):
        yield Static("╔═ Log ════════════════════════════════════════════════╗", classes="panel-header")
        yield self._log

    def on_mount(self) -> None:
        self._log.write("[dim]TUI started — waiting for events...[/]")

    def handle_pipeline_log(self, ev: PipelineLogEvent) -> None:
        style = _LEVEL_STYLE.get(ev.level, "")
        label = f"[{ev.level:<7}]"
        self._log.write(Text(f"{label} {ev.message}", style=style))

    def handle_training_log(self, ev: TrainingLogEvent) -> None:
        style = _LEVEL_STYLE.get(ev.level, "")
        label = f"[{ev.level:<7}]"
        self._log.write(Text(f"{label} {ev.message}", style=style))
