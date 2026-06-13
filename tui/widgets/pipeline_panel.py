# tui/widgets/pipeline_panel.py
"""Pipeline stage progress panel."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static, ListView, ListItem, Label
from textual.widget import Widget
from rich.table import Table
from rich.text import Text
from rich.style import Style
from typing import Optional

from tui.events import PipelineStageEvent, StageStatus


# ── Colour map ────────────────────────────────────────────────────────────

_STATUS_STYLE: dict[StageStatus, str] = {
    StageStatus.PENDING: "dim white",
    StageStatus.RUNNING: "bold cyan",
    StageStatus.DONE: "bold green",
    StageStatus.FAILED: "bold red",
    StageStatus.SKIPPED: "dim yellow",
}

_STATUS_ICON: dict[StageStatus, str] = {
    StageStatus.PENDING: "○",
    StageStatus.RUNNING: "◉",
    StageStatus.DONE: "●",
    StageStatus.FAILED: "✕",
    StageStatus.SKIPPED: "–",
}


# ── Stage row widget ─────────────────────────────────────────────────────

class StageRow(Widget):
    """A single pipeline stage row with icon, name, progress bar, status."""

    status: StageStatus = reactive(StageStatus.PENDING)
    progress: float = reactive(0.0)
    message: str = reactive("")

    def __init__(
        self, stage_name: str, status: StageStatus = StageStatus.PENDING
    ) -> None:
        super().__init__()
        self.stage_name = stage_name
        self.status = status

    def compose(self):
        yield Static(id=f"stage-{self.stage_name}", classes="stage-row")

    def watch_status(self, old: StageStatus, new: StageStatus) -> None:
        self._render()

    def watch_progress(self, old: float, new: float) -> None:
        self._render()

    def _render(self) -> None:
        icon = _STATUS_ICON.get(self.status, "○")
        style = _STATUS_STYLE.get(self.status, "dim white")
        name = self.stage_name.replace("_", " ").title()
        bar = self._progress_bar()
        t = Text()
        t.append(f" {icon} ", style)
        t.append(f"{name:24}", style=style)
        t.append(f" {bar}", style=style)
        if self.message:
            t.append(f"  {self.message}", style="dim white")
        self.query_one(f"#stage-{self.stage_name}", Static).update(t)

    def _progress_bar(self) -> str:
        w = 12
        filled = int(round(self.progress * w))
        empty = w - filled
        blocks = "█" * filled + "░" * empty
        pct = int(self.progress * 100)
        return f"[{blocks}] {pct:3d}%"


# ── Default stage list ───────────────────────────────────────────────────

CORE_STAGES = [
    "download",
    "sample",
    "augment",
    "create_ready",
    "validate",
    "vocabulary",
]


# ── Pipeline Panel ────────────────────────────────────────────────────────

class PipelinePanel(Widget):
    """Left panel showing all pipeline stages."""

    all_stages: dict[str, StageRow] = {}

    def __init__(self, stages: Optional[list[str]] = None) -> None:
        super().__init__(id="pipeline-panel")
        self._stage_names = stages or CORE_STAGES

    def compose(self):
        yield Static("╔═ Data Pipeline ═══╗", classes="panel-header")
        for name in self._stage_names:
            row = StageRow(name)
            self.all_stages[name] = row
            yield row
        yield Static("", id="pipeline-summary", classes="pipeline-summary")

    def on_mount(self) -> None:
        self._update_summary()

    def handle_stage_event(self, ev: PipelineStageEvent) -> None:
        row = self.all_stages.get(ev.stage)
        if row is None:
            # Unrecognised stage — create on the fly
            row = StageRow(ev.stage)
            self.all_stages[ev.stage] = row
            self.mount(row, before=self.query_one("#pipeline-summary"))
        row.status = ev.status
        row.progress = ev.progress
        row.message = ev.message
        self._update_summary()

    def _update_summary(self) -> None:
        done = sum(
            1 for r in self.all_stages.values() if r.status == StageStatus.DONE
        )
        total = len(self.all_stages)
        running = any(
            r.status == StageStatus.RUNNING for r in self.all_stages.values()
        )
        failed = any(
            r.status == StageStatus.FAILED for r in self.all_stages.values()
        )
        status = "RUNNING" if running else ("FAILED" if failed else "IDLE")
        self.query_one("#pipeline-summary", Static).update(
            f"  [{done}/{total}] stages complete — {status}"
        )
