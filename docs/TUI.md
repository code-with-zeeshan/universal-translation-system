# Terminal UI Dashboard

Real-time dashboard for monitoring data pipeline and training progress, built with [Textual](https://textual.textualize.io/).

## Quick Start

```bash
# Full pipeline → training (default)
uts tui --config config/base.yaml

# Data pipeline only
uts tui --config config/base.yaml --pipeline

# Training only
uts tui --config config/base.yaml --train
```

## Screen Layout

```
┌──────────────────────┬──────────────────────┐
│  Data Pipeline       │  Training            │
│  ○ download          │  Phase      training │
│  ○ sample            │  Epoch      3/5      │
│  ○ augment           │  Batch      124/5000 │
│  ○ create_ready      │  Loss       2.3456   │
│  ○ comet_quality     │  BLEU       28.40    │
│  ○ validate          │  LR         5.00e-04 │
│  ○ vocabulary        │  Tok/s      1420     │
│  [3/7] stages — …    │  GPU [████░░] 67%    │
├──────────────────────┴──────────────────────┤
│  Log                                         │
│  [INFO] Epoch 3/5 | batch 124/5000 | …       │
│  [INFO] Epoch 3/5 | train_loss 2.34 | …      │
└──────────────────────────────────────────────┘
```

### Panels

| Panel | Content |
|---|---|
| **Data Pipeline** (top-left) | Each pipeline stage with icon, name, progress bar, status message. Summary row shows `[done/total] stages complete` |
| **Training** (top-right) | Phase, epoch/batch progress, loss (color-coded), BLEU, learning rate, tokens/sec, GPU meter with utilization bar + memory |
| **Log** (bottom, full width) | Scrollable log window (500 lines max) showing INFO/WARNING/ERROR from both pipeline and trainer |

### Stage Indicators

| Icon | Status | Color |
|---|---|---|
| `●` | Done | Green |
| `◉` | Running | Cyan |
| `○` | Pending | Dim white |
| `✕` | Failed | Red |
| `–` | Skipped | Dim yellow |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Refresh all panels |
| `h` | Show help overlay |
| `Esc` | Close help overlay |

## Run Modes

- **Default** (no `--pipeline` or `--train`): Runs pipeline first, then automatically starts training when pipeline completes
- `--pipeline`: Runs data pipeline then stops
- `--train`: Runs training only (no pipeline phase, assumes data is ready)

## How It Works

The TUI uses three bridge classes that wrap existing components without modifying them:

| Component | Source | What It Does |
|---|---|---|
| `PipelineBridge` | `tui/bridge.py:145` | Wraps `UnifiedDataPipeline`, emits `PipelineStageEvent` / `PipelineLogEvent` for each stage |
| `TrainingBridge` | `tui/bridge.py:219` | Wraps `IntelligentTrainer`, emits `TrainingStartEvent` / `TrainingBatchEvent` / `TrainingEpochEvent` / `TrainingEvalEvent` / `TrainingDoneEvent` |
| `GPUMonitor` | `tui/bridge.py:278` | Polls NVML every 2s, emits `GPUStatusEvent` with utilization, memory, temperature |

### Event System (`tui/events.py`)

Events are dataclass-based and dispatched by type in `MainScreen.on_tui_event_message()`. Log lines are intercepted via custom `logging.Handler` subclasses that parse structured metrics (batch loss, epoch summary, BLEU) from log output.

### Panels (`tui/widgets/`)

| Widget | File | Description |
|---|---|---|
| `PipelinePanel` | `widgets/pipeline_panel.py` | Stage rows with reactive status/progress, dynamic stage creation |
| `TrainingPanel` | `widgets/training_panel.py` | Metric rows (loss, BLEU, LR, tok/s), GPU meter, ASCII loss history |
| `LogPanel` | `widgets/log_panel.py` | Scrolling RichLog, color-coded by log level |
| `GPUMeter` | `widgets/training_panel.py` | Bar + percentage + memory usage |

## File Structure

```
tui/
├── __init__.py         Package docstring
├── __main__.py         `python -m tui.app` entry point
├── app.py              TUIApp, MainScreen, CSS, background workers
├── bridge.py           PipelineBridge, TrainingBridge, GPUMonitor, log handlers
├── events.py           Event dataclasses (9 event types)
└── widgets/
    ├── __init__.py
    ├── log_panel.py         Scrolling log panel
    ├── pipeline_panel.py    Stage progress with reactive updates
    └── training_panel.py    Training metrics + GPU meter
```
