#!/usr/bin/env python3
"""
Interactive data pipeline stage selector.

Usage:
    python scripts/data_pipeline_wizard.py

Generates a YAML config override that can be passed to uts data --pipeline --config <path>.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent

# Stage: (name, display_name, description, default_enabled, time_min, notes)
STAGES: List[Tuple[str, str, str, bool, int, str]] = [
    # Core default stages
    ("download_training",  "download_training",   "OPUS-100 + extra sources",                        True,   30, ""),
    ("sample_filter",      "sample_filter",       "Deduplicate, filter length/content",              True,   10, ""),
    ("augment",            "augment",             "False friends, idioms, backtranslation, pivots",  True,   25, ""),
    ("create_ready",       "create_ready",        "Merge all sources → train_final.txt/val_final.txt", True, 5, ""),
    ("validate",           "validate",            "Validate output data and vocabulary files",       True,    3, ""),
    ("vocabulary",         "vocabulary",          "Build vocabulary packs from monolingual corpora", True,    8, ""),
    # Optional heavy stages
    ("wikipedia_backtranslation", "wikipedia_backtranslation", "Download Wikipedia monolingual data", False, 15, "internet"),
    ("direct_opus",        "direct_opus",          "Direct OPUS.nlpl.eu fallback download",          False,  20, "internet"),
    ("knowledge_distillation", "knowledge_distillation", "Distill NLLB-3.3B teacher into training data", False, 45, "GPU req."),
    ("download_evaluation", "download_evaluation", "Pre-fetch evaluation test sets",                 False,   2, "use: uts eval --download"),
    ("comet_quality",      "comet_quality",        "Neural quality filter (Unbabel/wmt22-comet-da)", False,  30, "GPU + 12GB VRAM"),
]


def fmt_time(mins: int) -> str:
    h, m = divmod(mins, 60)
    if h:
        return f"{h}h {m:02d}min"
    return f"{m} min"


def draw_menu(states: Dict[str, bool], cursor: int) -> None:
    os.system("clear" if os.name == "posix" else "cls")
    total = sum(s[4] for i, s in enumerate(STAGES) if states[s[0]])

    print("\033[1;36m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1;36m║       Data Pipeline Stage Selector                      ║\033[0m")
    print("\033[1;36m╚══════════════════════════════════════════════════════════╝\033[0m")
    print(f"  Use \033[1;33mUP/DOWN\033[0m to navigate, \033[1;33mSPACE\033[0m to toggle, \033[1;32mENTER\033[0m to confirm, \033[1;31mq\033[0m to quit")
    print()

    for i, (key, label, desc, default, mins, notes) in enumerate(STAGES):
        selected = states[key]
        check = "\033[1;32m✓\033[0m" if selected else "\033[1;31m✗\033[0m"
        prefix = " \033[1;33m▶\033[0m " if i == cursor else "   "

        if i == 0:
            print(f"  \033[1;37m── Default stages ──\033[0m")
        elif i == 6:
            print(f"  \033[1;37m── Optional heavy stages ──\033[0m")

        line = f"  {prefix} {check} \033[1;37m{i+1:>2}.\033[0m {label:<28} \033[90m{fmt_time(mins):>8}\033[0m  {desc}"
        if notes:
            line += f"  \033[90m({notes})\033[0m"
        if i == cursor:
            print(f"\033[7m{line}\033[0m")
        else:
            print(line)

    print()
    print(f"  \033[1;37mEstimated total time:\033[0m \033[1;33m{fmt_time(total)}\033[0m")
    print()


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


def get_choice(prompt: str, default: str = "") -> str:
    val = input(f"\033[1;37m{prompt}\033[0m" + (f" [\033[90m{default}\033[0m]: " if default else ": "))
    return val if val else default


def confirm(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    val = input(f"\033[1;37m{prompt}\033[0m [\033[90m{default_str}\033[0m]: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def generate_config(states: Dict[str, bool], output_path: str) -> str:
    enabled = [k for k, v in states.items() if v]
    config = {"pipeline": {"enabled_stages": enabled}}
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def preview_config(states: Dict[str, bool]) -> None:
    enabled = [k for k, v in states.items() if v]
    total = sum(s[4] for i, s in enumerate(STAGES) if states[s[0]])

    print("\n  \033[1;36m── Generated Configuration Preview ──\033[0m\n")
    print("  \033[1;37mpipeline:\033[0m")
    print("    \033[1;37menabled_stages:\033[0m")
    for s in enabled:
        print(f"      - \033[32m{s}\033[0m")
    print()
    print(f"  \033[1;37mEstimated total time: \033[1;33m{fmt_time(total)}\033[0m")
    print()


def edit_config_path(default_path: str) -> str:
    val = get_choice("Output config path", default_path)
    path = Path(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_wizard() -> int:
    states: Dict[str, bool] = {s[0]: s[3] for s in STAGES}
    cursor = 0

    try:
        while True:
            draw_menu(states, cursor)
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
                print("\n  \033[33mCancelled.\033[0m")
                return 1
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(STAGES):
                    cursor = idx
                    states[STAGES[idx][0]] = not states[STAGES[idx][0]]
    except KeyboardInterrupt:
        print("\n  \033[33mInterrupted.\033[0m")
        return 1

    # Confirm
    draw_menu(states, cursor)
    preview_config(states)
    if not confirm("Use this configuration?", True):
        print("  \033[33mCancelled.\033[0m")
        return 1

    # Output path
    default_path = str(ROOT / "config" / "generated_pipeline_config.json")
    output_path = edit_config_path(default_path)

    # Write
    generate_config(states, output_path)
    total = sum(s[4] for i, s in enumerate(STAGES) if states[s[0]])

    print(f"\n  \033[1;32m✓\033[0m Config written to \033[1;33m{output_path}\033[0m")
    print()
    print(f"  Run the pipeline with your custom stages:")
    print(f"  \033[1;37m$ uts data --pipeline --config {output_path}\033[0m")
    print(f"  \033[1;37m$ uts data --pipeline --config {output_path} --scale 5\033[0m")
    print()
    print(f"  \033[90mEstimated time: {fmt_time(total)}\033[0m")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(run_wizard())
