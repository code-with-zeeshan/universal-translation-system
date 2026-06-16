#!/usr/bin/env python3
"""
Interactive data pipeline stage selector.

Usage:
    python scripts/data_pipeline_wizard.py

Generates a JSON config override that can be passed to uts data --pipeline --config <path>.

Stage definitions and TUI helpers are in _wizard_shared.py (single source of truth).
"""

import json
import sys
from pathlib import Path
from typing import Dict

from scripts._wizard_shared import (
    STAGES,
    clear_screen,
    draw_stage_menu,
    fmt_time,
    get_key,
    run_stage_selector,
)


ROOT = Path(__file__).resolve().parent.parent


def confirm(prompt: str, default: bool = True) -> bool:
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not s else s.startswith("y")


def get_choice(prompt: str, default: str = "") -> str:
    val = input(f"{prompt}" + (f" [{default}]: " if default else ": "))
    return val if val else default


def generate_config(states: Dict[str, bool], output_path: str) -> str:
    enabled = [k for k, v in states.items() if v]
    config = {"pipeline": {"enabled_stages": enabled}}
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def preview_config(states: Dict[str, bool]) -> None:
    enabled = [k for k, v in states.items() if v]
    total = sum(s[4] for s in STAGES if states[s[0]])

    print("\n  \033[1;36m\u2500\u2500 Generated Configuration Preview \u2500\u2500\033[0m\n")
    print("  \033[1;37mpipeline:\033[0m")
    print("    \033[1;37menabled_stages:\033[0m")
    for s in enabled:
        print(f"      - \033[32m{s}\033[0m")
    print()
    print(f"  \033[1;37mEstimated total time: \033[1;33m{fmt_time(total)}\033[0m")
    print()


def run_wizard() -> int:
    default_stages = [s[0] for s in STAGES if s[3]]
    enabled = run_stage_selector(default_stages)
    if enabled is None:
        print("\n  \033[33mCancelled.\033[0m")
        return 1

    states = {s[0]: s[0] in enabled for s in STAGES}
    clear_screen()
    draw_stage_menu(states, cursor=-1)
    preview_config(states)

    if not confirm("Use this configuration?", True):
        print("  \033[33mCancelled.\033[0m")
        return 1

    default_path = str(ROOT / "config" / "generated_pipeline_config.json")
    output_path = get_choice("Output config path", default_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    generate_config(states, output_path)
    total = sum(s[4] for s in STAGES if states[s[0]])

    print(f"\n  \033[1;32m\u2713\033[0m Config written to \033[1;33m{output_path}\033[0m")
    print()
    print(f"  Run the pipeline with your custom stages:")
    print(f"  \033[1;37m$ uts data --pipeline --config {output_path}\033[0m")
    print(f"  \033[1;37m$ uts data --pipeline --config {output_path} --scale 5\033[0m")
    print()
    print(f"  \033[90mEstimated time: {fmt_time(total)}\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(run_wizard())
