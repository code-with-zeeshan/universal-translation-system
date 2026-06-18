"""Shared constants and TUI helpers for interactive config wizards.

Single source of truth for pipeline stage definitions, time estimates,
and terminal input helpers. Imported by both data_pipeline_wizard.py
and config_interactive.py.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional


# Stage: (key, label, description, default_enabled, time_minutes, notes, category)
# category: 'cpu' | 'gpu_light' | 'gpu_heavy'
#
# Quality stages (augment, knowledge_distillation) use NLLB-1.3B (distilled)
# to create meaning-preserving parallel data — they handle idioms, metaphors,
# and cultural equivalents rather than word-for-word swaps.
STAGES: List[Tuple[str, str, str, bool, int, str, str]] = [
    ("download_training",         "download_training",         "OPUS-100 + extra sources",
        True, 60, "", "cpu"),
    ("sample_filter",             "sample_filter",             "Deduplicate, length/content filter, quality scoring",
        True, 10, "", "cpu"),
    ("vocabulary",                "vocabulary",                "Build vocabulary packs from monolingual corpora",
        True, 10, "", "cpu"),
    ("create_ready",              "create_ready",              "Merge all sources -> train_final.txt/val_final.txt",
        True,  5, "", "cpu"),
    ("validate",                  "validate",                  "Validate output data and vocabulary files",
        True,  3, "", "cpu"),

    ("comet_quality",             "comet_quality",             "Neural semantic quality filter (Unbabel/wmt22-comet-da)",
        False, 90, "GPU ~1-2h on T4", "gpu_light"),

    ("augment",                   "augment",                   "Backtranslation + pivots for meaning-preserving synthetic pairs (NLLB-1.3B)",
        True, 1440, "GPU heavy (NLLB-1.3B)", "gpu_heavy"),
    ("knowledge_distillation",    "knowledge_distillation",    "NLLB-1.3B soft targets preserving semantic nuance over literal swaps",
        False, 2880, "GPU heavy (NLLB-1.3B)", "gpu_heavy"),
]

CATEGORY_LABELS = {
    "cpu": "CPU Stages (no GPU needed)",
    "gpu_light": "GPU-Light Stages (T4-friendly, ~1-2h)",
    "gpu_heavy": "Quality Stages (meaning-preserving data via NLLB-1.3B — heavy but essential)",
}

CATEGORY_COLORS = {
    "cpu": "\033[1;37m",        # white
    "gpu_light": "\033[1;32m",  # green
    "gpu_heavy": "\033[1;31m",  # red
}


def stage_category(key: str) -> str:
    for s in STAGES:
        if s[0] == key:
            return s[6]
    return "cpu"


def stages_by_category(cat: str) -> List[Tuple]:
    return [s for s in STAGES if s[6] == cat]


def fmt_time(mins: int) -> str:
    if mins >= 1440:
        d, h = divmod(mins, 1440)
        return f"{d}d {h}h"
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
    """Render the stage selector TUI with categories."""
    category_order = ["cpu", "gpu_light", "gpu_heavy"]

    flat_idx = 0
    total = 0
    color_reset = "\033[0m"
    color_dim = "\033[90m"

    print(f"\033[1;36m{'='*60}\033[0m")
    title = "Data Pipeline Stage Selector"
    print(f"\033[1;36m   {title:^54}\033[0m")
    print(f"\033[1;36m{'='*60}\033[0m")
    print("  UP/DOWN navigate \u00b7 SPACE toggle \u00b7 ENTER confirm \u00b7 q cancel\n")

    for cat in category_order:
        cat_stages = stages_by_category(cat)
        if not cat_stages:
            continue

        cat_label = CATEGORY_LABELS.get(cat, cat)
        cat_color = CATEGORY_COLORS.get(cat, "\033[1;37m")
        print(f"  {cat_color}\u2500\u2500 {cat_label} \u2500\u2500{color_reset}")

        for key, label, desc, default, mins, notes, category in cat_stages:
            selected = states[key]
            check = "\033[1;32m\u2713\033[0m" if selected else "\033[1;31m\u2717\033[0m"
            prefix = " \033[1;33m\u25b6\033[0m " if flat_idx == cursor else "   "

            if selected:
                total += mins

            line = f"  {prefix} {check} \033[1;37m{flat_idx+1:>2}.\033[0m {label:<28} {color_dim}{fmt_time(mins):>8}{color_reset}  {desc}"
            if notes:
                line += f"  {color_dim}({notes}){color_reset}"
            if flat_idx == cursor:
                print(f"\033[7m{line}\033[0m")
            else:
                print(line)

            flat_idx += 1

        print()

    print(f"  \033[1;37mEstimated total time:\033[0m \033[1;33m{fmt_time(total)}\033[0m")
    print()


def run_stage_selector(current_stages: list) -> Optional[List[str]]:
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
