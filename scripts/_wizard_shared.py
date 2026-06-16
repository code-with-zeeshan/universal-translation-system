"""Shared constants and TUI helpers for interactive config wizards.

Single source of truth for pipeline stage definitions, time estimates,
and terminal input helpers. Imported by both data_pipeline_wizard.py
and config_interactive.py.
"""

import os
import sys
from typing import Dict, List, Tuple


# Stage: (key, label, description, default_enabled, time_minutes, notes)
STAGES: List[Tuple[str, str, str, bool, int, str]] = [
    ("download_training",         "download_training",         "OPUS-100 + extra sources",
        True, 30, ""),
    ("sample_filter",             "sample_filter",             "Deduplicate, filter length/content",
        True, 10, ""),
    ("augment",                   "augment",                   "False friends, idioms, backtranslation, pivots",
        True, 25, ""),
    ("create_ready",              "create_ready",              "Merge all sources -> train_final.txt/val_final.txt",
        True,  5, ""),
    ("validate",                  "validate",                  "Validate output data and vocabulary files",
        True,  3, ""),
    ("vocabulary",                "vocabulary",                "Build vocabulary packs from monolingual corpora",
        True,  8, ""),
    ("wikipedia_backtranslation", "wikipedia_backtranslation", "Download Wikipedia monolingual data",
        False, 15, "internet"),
    ("direct_opus",               "direct_opus",               "Direct OPUS.nlpl.eu fallback download",
        False, 20, "internet"),
    ("knowledge_distillation",    "knowledge_distillation",    "Distill NLLB-3.3B teacher into training data",
        False, 45, "GPU req."),
    ("download_evaluation",       "download_evaluation",       "Pre-fetch evaluation test sets",
        False,  2, "use: uts eval --download"),
    ("comet_quality",             "comet_quality",             "Neural quality filter (Unbabel/wmt22-comet-da)",
        False, 30, "GPU + 12GB VRAM"),
]


def fmt_time(mins: int) -> str:
    h, m = divmod(mins, 60)
    return f"{h}h {m:02d}min" if h else f"{m} min"


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def get_key() -> str:
    """Read a single keypress. Falls back to input() when stdin is not a TTY."""
    if not sys.stdin.isatty():
        return input("Enter number to toggle, ENTER to confirm, q to quit: ").strip()
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(2)
            if nxt == "[A":
                return "UP"
            elif nxt == "[B":
                return "DOWN"
            return "ESC"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def draw_stage_menu(states: Dict[str, bool], cursor: int):
    """Render the stage selector TUI."""
    total = sum(s[4] for i, s in enumerate(STAGES) if states[s[0]])

    print("\033[1;36m\u2500\u2500 Data Pipeline Stage Selector \u2500\u2500\033[0m")
    print("  UP/DOWN navigate \u00b7 SPACE toggle \u00b7 ENTER confirm \u00b7 q cancel\n")

    for i, (key, label, desc, default, mins, notes) in enumerate(STAGES):
        selected = states[key]
        check = "\033[1;32m\u2713\033[0m" if selected else "\033[1;31m\u2717\033[0m"
        prefix = " \033[1;33m\u25b6\033[0m " if i == cursor else "   "

        if i == 0:
            print("  \033[1;37m\u2500\u2500 Default stages \u2500\u2500\033[0m")
        elif i == 6:
            print("  \033[1;37m\u2500\u2500 Optional heavy stages \u2500\u2500\033[0m")

        line = f"  {prefix} {check} \033[1;37m{i+1:>2}.\033[0m {label:<28} \033[90m{fmt_time(mins):>8}\033[0m  {desc}"
        if notes:
            line += f"  \033[90m({notes})\033[0m"
        if i == cursor:
            print(f"\033[7m{line}\033[0m")
        else:
            print(line)

    print(f"\n  \033[1;37mEstimated total time:\033[0m \033[1;33m{fmt_time(total)}\033[0m")


def run_stage_selector(current_stages: list) -> list | None:
    """Run the interactive TUI stage selector.

    Args:
        current_stages: List of currently enabled stage keys (from config).

    Returns:
        List of enabled stage keys, or None if cancelled.
    """
    states: Dict[str, bool] = {s[0]: s[0] in current_stages for s in STAGES}
    cursor = 0

    if not sys.stdin.isatty():
        print("Pipeline stages (comma-separated, ENTER to keep):")
        print(", ".join(s[0] for s in STAGES))
        val = input("Stages: ").strip()
        if val:
            return [x.strip() for x in val.split(",") if x.strip()]
        return current_stages

    try:
        while True:
            clear_screen()
            draw_stage_menu(states, cursor)
            key = get_key()
            if key == "UP":
                cursor = (cursor - 1) % len(STAGES)
            elif key == "DOWN":
                cursor = (cursor + 1) % len(STAGES)
            elif key == " ":
                states[STAGES[cursor][0]] = not states[STAGES[cursor][0]]
            elif key in ("\r", "\n", ""):
                break
            elif key.lower() == "q":
                return None
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(STAGES):
                    cursor = idx
                    states[STAGES[idx][0]] = not states[STAGES[idx][0]]
    except KeyboardInterrupt:
        return None

    return [k for k, v in states.items() if v]
