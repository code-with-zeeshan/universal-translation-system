# tui/widgets/training_panel.py
"""Training metrics panel — epoch, loss, BLEU, GPU, LR."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static, RichLog
from textual.widget import Widget
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from typing import Optional

from tui.events import (
    GPUStatusEvent,
    TrainingBatchEvent,
    TrainingDoneEvent,
    TrainingEpochEvent,
    TrainingEvalEvent,
    TrainingStartEvent,
)


# ── GPU Meter ─────────────────────────────────────────────────────────────

class GPUMeter(Widget):
    """GPU utilisation bar + memory."""

    gpu_util: float = reactive(0.0)
    mem_used_gb: float = reactive(0.0)
    mem_total_gb: float = reactive(0.0)

    def compose(self):
        yield Static(id="gpu-display")

    def watch_gpu_util(self) -> None:
        self._render()

    def _render(self) -> None:
        w = 20
        fill = int(round(self.gpu_util / 100 * w))
        bar = "█" * fill + "░" * (w - fill)
        col = "green" if self.gpu_util < 50 else ("yellow" if self.gpu_util < 80 else "red")
        mem = f"{self.mem_used_gb:.1f}/{self.mem_total_gb:.1f} GB"
        t = Text()
        t.append(f"GPU  [{bar}] ", style=f"bold {col}")
        t.append(f"{self.gpu_util:5.1f}%  ", style=f"{col}")
        t.append(mem, style="dim")
        self.query_one("#gpu-display", Static).update(t)


# ── Metric row ────────────────────────────────────────────────────────────

class MetricRow(Widget):
    """Single labeled value."""

    def __init__(self, label: str, value: str = "—", color: str = "white") -> None:
        super().__init__()
        self._label = label
        self._value = value
        self._color = color

    def compose(self):
        yield Static(id=f"metric-{self._label.lower().replace(' ', '-')}")

    def update(self, value: str, color: str = "white") -> None:
        self._value = value
        self._color = color
        t = Text()
        t.append(f"  {self._label:14}", style="bold")
        t.append(f"{value}", style=color)
        self.query_one(Static).update(t)

    def on_mount(self) -> None:
        self.update(self._value, self._color)


# ── Training Panel ────────────────────────────────────────────────────────

class TrainingPanel(Widget):
    """Right panel showing live training metrics."""

    def __init__(self) -> None:
        super().__init__(id="training-panel")
        self._epoch: int = 0
        self._total_epochs: int = 0
        self._train_loss: float = 0.0
        self._val_loss: Optional[float] = None
        self._bleu: float = 0.0
        self._lr: float = 0.0
        self._tok_s: float = 0.0
        self._batch: int = 0
        self._total_batches: int = 0
        self._phase: str = "idle"
        self._history: list[float] = []
        self._gpu = GPUMeter()

    def compose(self):
        yield Static("╔═ Training ════════╗", classes="panel-header")
        self._phase_row = MetricRow("Phase", "idle", "dim")
        yield self._phase_row
        self._epoch_row = MetricRow("Epoch", "—")
        yield self._epoch_row
        self._batch_row = MetricRow("Batch", "—")
        yield self._batch_row
        self._loss_row = MetricRow("Loss", "—")
        yield self._loss_row
        self._bleu_row = MetricRow("BLEU", "—")
        yield self._bleu_row
        self._lr_row = MetricRow("LR", "—")
        yield self._lr_row
        self._tok_row = MetricRow("Tok/s", "—")
        yield self._tok_row
        yield self._gpu
        yield Static("", id="training-history", classes="training-history")

    def handle_start(self, ev: TrainingStartEvent) -> None:
        self._total_epochs = ev.total_epochs
        self._phase = "training"
        self._phase_row.update("training", "bold cyan")
        self._epoch_row.update(f"0/{ev.total_epochs}")
        self._lr_row.update(f"{ev.learning_rate:.2e}")
        self._batch_row.update(f"0/—")

    def handle_batch(self, ev: TrainingBatchEvent) -> None:
        self._train_loss = ev.loss
        self._lr = ev.learning_rate
        self._tok_s = ev.tokens_per_second
        self._batch = ev.batch
        self._total_batches = ev.total_batches
        self._epoch = ev.epoch
        self._render_batch()

    def handle_epoch(self, ev: TrainingEpochEvent) -> None:
        self._train_loss = ev.train_loss
        self._val_loss = ev.val_loss
        self._lr = ev.learning_rate
        self._tok_s = ev.tokens_per_second
        self._epoch = ev.epoch
        self._history.append(ev.train_loss)
        self._render_epoch()

    def handle_eval(self, ev: TrainingEvalEvent) -> None:
        self._bleu = ev.bleu
        self._val_loss = ev.val_loss
        self._bleu_row.update(f"{ev.bleu:.2f}", "green")
        self._history_line()

    def handle_done(self, ev: TrainingDoneEvent) -> None:
        self._phase = "done"
        self._phase_row.update("done", "bold green")
        m, s = divmod(int(ev.total_time_s), 60)
        h, m = divmod(m, 60)
        duration = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        self.query_one("#training-history", Static).update(
            f"\n  Finished in {duration} — best BLEU: {ev.best_bleu:.2f} (epoch {ev.best_epoch})"
        )

    def handle_gpu(self, ev: GPUStatusEvent) -> None:
        self._gpu.gpu_util = ev.gpu_util
        self._gpu.mem_used_gb = ev.memory_used_gb
        self._gpu.mem_total_gb = ev.memory_total_gb

    def _render_batch(self) -> None:
        self._epoch_row.update(f"{self._epoch}/{self._total_epochs}")
        self._batch_row.update(f"{self._batch}/{self._total_batches}")
        c = "red" if self._train_loss > 5 else ("yellow" if self._train_loss > 2 else "green")
        self._loss_row.update(f"{self._train_loss:.4f}", c)
        self._lr_row.update(f"{self._lr:.2e}")
        self._tok_row.update(f"{self._tok_s:.0f}")

    def _render_epoch(self) -> None:
        self._epoch_row.update(f"{self._epoch}/{self._total_epochs}", "bold cyan")
        c = "red" if self._train_loss > 5 else ("yellow" if self._train_loss > 2 else "green")
        self._loss_row.update(f"{self._train_loss:.4f}", c)
        self._lr_row.update(f"{self._lr:.2e}")
        self._tok_row.update(f"{self._tok_s:.0f}")

    def _history_line(self) -> None:
        """Render a simple ASCII loss curve."""
        if not self._history:
            return
        # Show last 8 epochs as a sparkline-ish bar
        recent = self._history[-8:]
        max_v = max(recent) or 1
        lines = []
        for val in recent:
            h = max(1, int(val / max_v * 8))
            lines.append("█" * h)
        self.query_one("#training-history", Static).update(
            f"  Loss history (last {len(recent)} epochs):\n  " + "  ".join(lines)
        )
