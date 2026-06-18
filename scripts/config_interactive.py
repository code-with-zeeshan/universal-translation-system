#!/usr/bin/env python3
"""
config_interactive.py — Interactive config builder.

Loads config/base.yaml as defaults, lets you override pipeline stages,
training settings, model architecture, and data settings interactively,
then saves the complete merged config to config/override/<name>.yaml.

Stage definitions and TUI helpers are in _wizard_shared.py (single source of truth).

Usage:
    python scripts/config_interactive.py                # Interactive TUI
    python scripts/config_interactive.py --set training.num_epochs=20  # CLI
    python scripts/config_interactive.py --non-interactive --set ...   # Batch
    python scripts/config_interactive.py --list                        # List overrides
    python scripts/config_interactive.py --diff my_config              # Diff vs base
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from scripts._wizard_shared import clear_screen, fmt_time, run_stage_selector


ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config" / "base.yaml"
OVERRIDE_DIR = ROOT / "config" / "override"


# ── Helpers ──────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    result = {}
    for key in list(base) + [k for k in override if k not in base]:
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = deep_merge(base[key], override[key])
            else:
                result[key] = override[key]
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def save_config(config: dict, name: str) -> Path:
    OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
    path = OVERRIDE_DIR / f"{name}.yaml"

    # Reorder top-level keys to match base.yaml style
    key_order = ["project", "model", "training", "data", "pipeline", "hub", "memory"]
    ordered = {k: config[k] for k in key_order if k in config}
    for k in config:
        if k not in ordered:
            ordered[k] = config[k]

    with open(path, "w") as f:
        yaml.dump(ordered, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def dot_set(config: dict, key: str, value: Any):
    parts = key.split(".")
    d = config
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_value(val: str) -> Any:
    v = val.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ── TUI Input Helpers ───────────────────────────────────────────────

def get_input(prompt: str, default: str = "") -> str:
    val = input(f"{prompt}" + (f" [{default}]: " if default else ": "))
    return val if val else default


def get_bool(prompt: str, default: bool = False) -> bool:
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not s else s.startswith("y")


def get_int(prompt: str, default: int, min_v: int = 0, max_v: int = 10**9) -> int:
    while True:
        v = input(f"{prompt} [{default}]: ").strip()
        if not v:
            return default
        try:
            n = int(v)
            if n < min_v:
                print(f"  Minimum: {min_v}")
                continue
            if n > max_v:
                print(f"  Maximum: {max_v}")
                continue
            return n
        except ValueError:
            print("  Enter a valid integer.")


def get_float(prompt: str, default: float) -> float:
    while True:
        v = input(f"{prompt} [{default}]: ").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            print("  Enter a valid number.")


def wait_enter():
    input("\n  Press ENTER to continue...")


# ── Section: Training Settings ───────────────────────────────────────

def section_training(config: dict) -> dict:
    t = config.get("training", {})
    overrides = {}
    print("\n\u2500\u2500 Training Settings (ENTER = keep current) \u2500\u2500\n")
    for field, label, getter in [
        ("batch_size", "Batch size", lambda: get_int("Batch size", t.get("batch_size", 32), 1)),
        ("num_epochs", "Epochs", lambda: get_int("Epochs", t.get("num_epochs", 10), 1)),
        ("lr", "Learning rate", lambda: get_float("Learning rate", t.get("lr", 3e-4))),
        ("warmup_steps", "Warmup steps", lambda: get_int("Warmup steps", t.get("warmup_steps", 1000), 0)),
        ("weight_decay", "Weight decay", lambda: get_float("Weight decay", t.get("weight_decay", 0.01))),
        ("accumulation_steps", "Accumulation steps", lambda: get_int("Accumulation steps", t.get("accumulation_steps", 4), 1)),
    ]:
        val = getter()
        if val != t.get(field):
            overrides[field] = val
    mixed = get_bool("Mixed precision", t.get("mixed_precision", True))
    if mixed != t.get("mixed_precision"):
        overrides["mixed_precision"] = mixed
    use_lora = get_bool("Use LoRA", t.get("use_lora", False))
    if use_lora != t.get("use_lora"):
        overrides["use_lora"] = use_lora
    if use_lora:
        lr = get_int("LoRA rank (encoder)", t.get("lora_r", 16), 1)
        if lr != t.get("lora_r"):
            overrides["lora_r"] = lr
        lrd = get_int("LoRA rank (decoder)", t.get("lora_r_decoder", 64), 1)
        if lrd != t.get("lora_r_decoder"):
            overrides["lora_r_decoder"] = lrd
    return {"training": overrides} if overrides else {}


# ── Section: Model Architecture ──────────────────────────────────────

def section_model(config: dict) -> dict:
    m = config.get("model", {})
    overrides = {}
    print("\n\u2500\u2500 Model Architecture (ENTER = keep current) \u2500\u2500\n")
    for field, label, default, getter in [
        ("hidden_dim", "Hidden dimension", 512, lambda d: get_int(label, d, 64)),
        ("num_layers", "Encoder layers", 6, lambda d: get_int(label, d, 1)),
        ("num_heads", "Encoder heads", 8, lambda d: get_int(label, d, 1)),
        ("decoder_dim", "Decoder dimension", 512, lambda d: get_int(label, d, 64)),
        ("decoder_layers", "Decoder layers", 8, lambda d: get_int(label, d, 1)),
        ("decoder_heads", "Decoder heads", 8, lambda d: get_int(label, d, 1)),
    ]:
        val = getter(m.get(field, default))
        if val != m.get(field, default):
            overrides[field] = val
    dropout = get_float("Dropout", m.get("dropout", 0.1))
    if abs(dropout - m.get("dropout", 0.1)) > 1e-10:
        overrides["dropout"] = dropout
    return {"model": overrides} if overrides else {}


# ── Section: Data Settings ───────────────────────────────────────────

def section_data(config: dict) -> dict:
    d = config.get("data", {})
    overrides = {}
    print("\n\u2500\u2500 Data Settings (ENTER = keep current) \u2500\u2500\n")
    max_len = get_int("Max sentence length", d.get("max_sentence_length", 128), 16)
    if max_len != d.get("max_sentence_length"):
        overrides["max_sentence_length"] = max_len
    seed = get_int("Random seed", d.get("seed", 42), 0)
    if seed != d.get("seed"):
        overrides["seed"] = seed
    current_langs = d.get("languages", [])
    print(f"\nCurrent languages ({len(current_langs)}): {', '.join(current_langs)}")
    if get_bool("Change language list?", False):
        lang_str = get_input("Languages (comma-sep)", ",".join(current_langs))
        new_langs = [l.strip() for l in lang_str.split(",") if l.strip()]
        if new_langs != current_langs:
            overrides["languages"] = new_langs
    return {"data": overrides} if overrides else {}


# ── Section: Advanced Data Quality Settings ──────────────────────────

def section_data_quality(config: dict) -> dict:
    print("\n\u2500\u2500 Advanced Data Quality (ENTER = keep current) \u2500\u2500\n")
    overrides = {}

    td = config.get("data", {}).get("training_distribution", {})
    print(f"  Current training_distribution: {len(td)} language pairs")
    if get_bool("Adjust per-pair sentence counts?", False):
        new_td = {}
        print("  Enter count for each pair (blank = keep current, 0 = remove):")
        for pair in sorted(td.keys()):
            cur = td[pair]
            val = get_input(f"    {pair}", str(cur))
            if val:
                try:
                    n = int(val)
                    if n > 0:
                        new_td[pair] = n
                except ValueError:
                    new_td[pair] = cur
        if new_td != td:
            overrides.setdefault("data", {})["training_distribution"] = new_td

    ap = config.get("data", {}).get("augmentation_pairs", [])
    print(f"\n  Current augmentation_pairs: {len(ap)} pairs")
    if get_bool("Change augmentation pairs?", False):
        ap_str = get_input("Augmentation pairs (comma-sep, e.g. en-es,fr-de)", ",".join(ap))
        new_ap = [p.strip() for p in ap_str.split(",") if p.strip()]
        if new_ap != ap:
            overrides.setdefault("data", {})["augmentation_pairs"] = new_ap

    vs = config.get("data", {}).get("vocabulary_strategy", {})
    print(f"\n  Vocab approach: {vs.get('approach', 'production')}")
    if get_bool("Change vocab strategy?", False):
        approach = get_input("  Vocab approach (production/research)", vs.get("approach", "production"))
        if approach.lower() in ("production", "research"):
            overrides.setdefault("data", {}).setdefault("vocabulary_strategy", {})["approach"] = approach.lower()
        groups = vs.get("groups", {})
        print(f"  Current vocab groups: {len(groups)} (latin, cjk, arabic, ...)")
        if get_bool("Edit vocab groups?", False):
            print("  Enter groups as: group_name: lang1,lang2,...  (blank line to finish):")
            new_groups = {}
            while True:
                line = input("    ").strip()
                if not line:
                    break
                if ":" in line:
                    gname, langs = line.split(":", 1)
                    new_groups[gname.strip()] = [l.strip() for l in langs.split(",") if l.strip()]
            if new_groups:
                overrides.setdefault("data", {}).setdefault("vocabulary_strategy", {})["groups"] = new_groups

    ds = config.get("data_strategy", {})
    print(f"\n  Current data_strategy priority_rules: high={len(ds.get('priority_rules', {}).get('high', []))} pairs")
    if get_bool("Change data strategy priority rules?", False):
        ds_override = {}
        for tier in ("high", "medium", "low"):
            cur = ds.get("priority_rules", {}).get(tier, [])
            if get_bool(f"  Edit {tier} priority pairs?", False):
                new_list = get_input(f"  {tier} pairs (comma-sep)", ",".join(cur))
                ds_override.setdefault("priority_rules", {})[tier] = [p.strip() for p in new_list.split(",") if p.strip()]
        ds_config = config.get("data_strategy", {})
        sp = ds_config.get("source_preferences", {})
        print("  Source preferences control which datasets are used per language group.")
        if get_bool("Change source preferences?", False):
            pref_override = {}
            for group, sources in sp.items():
                cur_str = ",".join(sources)
                new_str = get_input(f"  {group} sources (comma-sep)", cur_str)
                if new_str != cur_str:
                    pref_override[group] = [s.strip() for s in new_str.split(",") if s.strip()]
            if pref_override:
                ds_override["source_preferences"] = pref_override
        if ds_override:
            overrides["data_strategy"] = ds_override

    ct = config.get("pipeline", {}).get("comet_quality_threshold", 0.7)
    new_ct = get_float("Comet quality threshold (0.0-1.0)", ct)
    if abs(new_ct - ct) > 1e-10 and 0 <= new_ct <= 1:
        overrides.setdefault("pipeline", {})["comet_quality_threshold"] = new_ct

    ff = config.get("pipeline", {}).get("max_dynamic_ff_per_pair", 25000)
    new_ff = get_int("Max false friends per pair", ff, 0)
    if new_ff != ff:
        overrides.setdefault("pipeline", {})["max_dynamic_ff_per_pair"] = new_ff

    return overrides


# ── Section: Review & Save ──────────────────────────────────────────

def section_review(base: dict, merged: dict) -> Path | None:
    clear_screen()
    print("\033[1;36m\u2500\u2500 Review & Save Configuration \u2500\u2500\033[0m\n")
    changes = _compute_diff(base, merged)
    if changes:
        print("  \033[1;33mChanges from base.yaml:\033[0m\n")
        for path, old, new in changes:
            print(f"  \033[32m\u2022 {path}\033[0m")
            print(f"    \033[90m  base: {old}\033[0m")
            print(f"    \033[1;37m  new:  {new}\033[0m")
    else:
        print("  \033[90mNo changes from base.yaml.\033[0m\n")
    print("  \033[1;37mFull configuration:\033[0m\n")
    dumped = yaml.dump(merged, default_flow_style=False, sort_keys=False)
    print(dumped[:2500])
    if len(dumped) > 2500:
        print("  \033[90m... (truncated)\033[0m")
    print()
    if not get_bool("Save this configuration?", True):
        return None
    name = get_input("Config name", "custom") or "custom"
    path = save_config(merged, name)
    print(f"\n  \033[1;32m\u2713\033[0m Saved to \033[1;33m{path}\033[0m")
    print(f"\n  Usage:")
    print(f"    \033[1;37m$ uts data --pipeline --config {path}\033[0m")
    print(f"    \033[1;37m$ uts train --full --config {path}\033[0m")
    print(f"    \033[1;37m$ uts tui --config {path}\033[0m")
    return path


def _compute_diff(base: dict, merged: dict, prefix: str = "") -> List[Tuple[str, Any, Any]]:
    changes = []
    for key in sorted(set(base) | set(merged)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            changes.append((path, "<missing>", merged[key]))
        elif key not in merged:
            changes.append((path, base[key], "<missing>"))
        elif isinstance(base[key], dict) and isinstance(merged[key], dict):
            changes.extend(_compute_diff(base[key], merged[key], path))
        elif base[key] != merged[key]:
            changes.append((path, base[key], merged[key]))
    return changes


# ── CLI override application ─────────────────────────────────────────

def apply_cli_overrides(base: dict, sets: List[str]) -> dict:
    merged = deep_merge({}, base)
    for s in sets:
        if "=" not in s:
            print(f"  \033[31mInvalid --set: {s} (expected key=value)\033[0m")
            continue
        key, val = s.split("=", 1)
        v = parse_value(val)
        dot_set(merged, key, v)
        print(f"  \033[32m\u2022 {key} = {repr(v)}\033[0m")
    return merged


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive config builder for Universal Translation System",
        epilog=(
            "Examples:\n"
            "  python scripts/config_interactive.py\n"
            "  python scripts/config_interactive.py --set training.num_epochs=20\n"
            "  python scripts/config_interactive.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a setting (dot notation)")
    parser.add_argument("--name", default="custom",
                        help="Output config name (default: custom)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip TUI, just apply --set overrides and save")
    parser.add_argument("--list", action="store_true",
                        help="List existing override configs")
    parser.add_argument("--diff", metavar="NAME",
                        help="Show diff between override and base.yaml")
    args = parser.parse_args()

    if not BASE_CONFIG.exists():
        print(f"Error: {BASE_CONFIG} not found")
        return 1

    base = load_yaml(BASE_CONFIG)

    if args.list:
        files = sorted(OVERRIDE_DIR.glob("*.yaml")) if OVERRIDE_DIR.exists() else []
        if not files:
            print("No override configs found in", OVERRIDE_DIR)
            return 0
        print("Override configs:")
        for f in files:
            print(f"  \033[1;33m{f.stem}\033[0m  ({f})")
        return 0

    if args.diff:
        path = OVERRIDE_DIR / f"{args.diff}.yaml"
        if not path.exists():
            path = Path(args.diff)
        if not path.exists():
            print(f"Error: {args.diff} not found")
            return 1
        override = load_yaml(path)
        changes = _compute_diff(base, override)
        if not changes:
            print("No differences from base.yaml.")
            return 0
        print(f"Differences: {path}")
        for path, old, new in changes:
            print(f"  \033[32m\u2022 {path}\033[0m")
            print(f"    \033[90m  base: {old}\033[0m")
            print(f"    \033[1;37m  new:  {new}\033[0m")
        return 0

    merged = apply_cli_overrides(base, args.set)

    if args.non_interactive or not sys.stdin.isatty():
        path = save_config(merged, args.name)
        print(f"\nSaved to {path}")
        return 0

    clear_screen()
    print("\033[1;36m\u2500\u2500 Universal Translation System \u2014 Interactive Config Builder \u2500\u2500\033[0m")
    print()
    print("  This wizard helps you create a config based on base.yaml.")
    print("  Each section shows defaults; press ENTER to keep.")
    print("  Sections: 1) Pipeline Stages  2) Training  3) Model  4) Data  5) Advanced Quality  6) Review")
    wait_enter()

    # Section 1: Pipeline Stages
    clear_screen()
    print("Section 1/6: Pipeline Stages\n")
    current_stages = merged.get("pipeline", {}).get("enabled_stages", [])
    enabled = run_stage_selector(current_stages)
    if enabled is None:
        print("Cancelled.")
        return 1
    merged.setdefault("pipeline", {})["enabled_stages"] = enabled
    wait_enter()

    # Section 2: Training
    clear_screen()
    print("Section 2/6: Training Settings\n")
    tr = section_training(merged)
    if tr:
        merged = deep_merge(merged, tr)
    wait_enter()

    # Section 3: Model
    clear_screen()
    print("Section 3/6: Model Architecture\n")
    mo = section_model(merged)
    if mo:
        merged = deep_merge(merged, mo)
    wait_enter()

    # Section 4: Data
    clear_screen()
    print("Section 4/6: Data Settings\n")
    da = section_data(merged)
    if da:
        merged = deep_merge(merged, da)
    wait_enter()

    # Section 4b: Advanced Data Quality (optional)
    clear_screen()
    print("Section 5/6: Advanced Data Quality Settings\n")
    print("  This section controls data quality levers: training_distribution,")
    print("  augmentation_pairs, vocab strategy groups, data strategy priorities,")
    print("  comet threshold, and false friend limits.")
    if get_bool("Configure advanced data quality settings?", True):
        dq = section_data_quality(merged)
        if dq:
            merged = deep_merge(merged, dq)
    wait_enter()

    # Section 6: Review & Save
    result = section_review(base, merged)
    if result is None:
        print("Cancelled.")
        return 1
    print(f"\n  \033[1;32mDone!\033[0m Configuration saved to \033[1;33m{result}\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
