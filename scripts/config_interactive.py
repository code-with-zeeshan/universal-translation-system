#!/usr/bin/env python3
"""
config_interactive.py — Interactive config builder.

Loads config/base.yaml as defaults, lets you override all settings
interactively, then saves complete merged config to config/override/<name>.yaml.

Usage:
    python scripts/config_interactive.py                           # Interactive TUI
    python scripts/config_interactive.py --preset-stages a,b,c     # Pre-fill stages
    python scripts/config_interactive.py --set training.num_epochs=20  # CLI batch
    python scripts/config_interactive.py --non-interactive --set ...   # Batch
    python scripts/config_interactive.py --list                    # List overrides
    python scripts/config_interactive.py --diff my_config          # Diff vs base
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from scripts._wizard_shared import clear_screen, fmt_time, run_stage_selector


ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config" / "base.yaml"
OVERRIDE_DIR = ROOT / "config" / "override"

# Base.yaml key order for output
KEY_ORDER = [
    "config_version", "data_version", "pipeline_version",
    "data", "data_strategy", "model", "training", "distributed",
    "monitoring", "pipeline", "security", "hub",
]

TW = "\033[1;33m[TWEAK]\033[0m"
AD = "\033[1;35m[ADVANCED]\033[0m"
OP = "\033[1;32m[OPTIONAL]\033[0m"


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
    ordered = {k: config[k] for k in KEY_ORDER if k in config}
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
    if v.lower() == "true": return True
    if v.lower() == "false": return False
    if v.lower() == "none": return None
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v


def get_input(prompt: str, default: str = "") -> str:
    val = input(f"{prompt}" + (f" [{default}]: " if default else ": "))
    return val if val else default


def get_bool(prompt: str, default: bool = False) -> bool:
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not s else s.startswith("y")


def get_int(prompt: str, default: int, min_v: int = 0, max_v: int = 10**9) -> int:
    while True:
        v = input(f"{prompt} [{default}]: ").strip()
        if not v: return default
        try:
            n = int(v)
            if n < min_v: print(f"  Minimum: {min_v}"); continue
            if n > max_v: print(f"  Maximum: {max_v}"); continue
            return n
        except ValueError: print("  Enter a valid integer.")


def get_float(prompt: str, default: float) -> float:
    while True:
        v = input(f"{prompt} [{default}]: ").strip()
        if not v: return default
        try: return float(v)
        except ValueError: print("  Enter a valid number.")


def get_str_list(prompt: str, default: List[str]) -> List[str]:
    val = input(f"{prompt} [{','.join(default)}]: ").strip()
    if not val: return default
    return [x.strip() for x in val.split(",") if x.strip()]


def wait_enter():
    input("\n  Press ENTER to continue...")


def _section_header(num: int, total: int, title: str, desc: str = ""):
    clear_screen()
    print(f"\033[1;36mSection {num}/{total}: {title}\033[0m")
    if desc:
        print(f"  \033[90m{desc}\033[0m")
    print()


# ── Section 1: Pipeline Stages ──────────────────────────────────────

def section_stages(base: dict, preset_stages: List[str] = None) -> dict:
    _section_header(1, 9, "Pipeline Stages",
                    "Select which stages to run. ESSENTIAL = must-run, "
                    "RECOMMENDED = improves quality, ADVANCED = best but expensive")
    current = preset_stages or base.get("pipeline", {}).get("enabled_stages", [])
    enabled = run_stage_selector(current)
    if enabled is None:
        return None
    return {"pipeline": {"enabled_stages": enabled}}


# ── Section 2: Training ─────────────────────────────────────────────

def section_training(config: dict) -> dict:
    _section_header(2, 9, "Training Settings",
                    f"{TW} = commonly tweaked  |  {AD} = leave as default for most users")
    t = config.get("training", {})
    overrides = {}

    for field, label, hint, getter in [
        ("num_epochs", "Epochs", TW, lambda: get_int("  Epochs", t.get("num_epochs", 10), 1)),
        ("batch_size", "Batch size per GPU", TW, lambda: get_int("  Batch size", t.get("batch_size", 32), 1)),
        ("accumulation_steps", "Gradient accumulation steps", TW,
         lambda: get_int("  Accumulation steps", t.get("accumulation_steps", 4), 1)),
        ("lr", "Learning rate", TW, lambda: get_float("  Learning rate", t.get("lr", 3e-4))),
        ("warmup_steps", "Warmup steps", TW, lambda: get_int("  Warmup steps", t.get("warmup_steps", 1000), 0)),
        ("weight_decay", "Weight decay", AD, lambda: get_float("  Weight decay", t.get("weight_decay", 0.01))),
    ]:
        print(f"  {hint} {label}")
        val = getter()
        if val != t.get(field):
            overrides[field] = val

    print()
    use_fsdp = get_bool(f"  {AD} Use FSDP (Fully Sharded Data Parallel)", t.get("use_fsdp", False))
    if use_fsdp != t.get("use_fsdp"):
        overrides["use_fsdp"] = use_fsdp

    mp = get_bool(f"  {TW} Mixed precision", t.get("mixed_precision", True))
    if mp != t.get("mixed_precision"):
        overrides["mixed_precision"] = mp

    dtype = get_input(f"  {AD} Dtype (bfloat16/float16/float32)", t.get("dtype", "bfloat16"))
    if dtype != t.get("dtype"):
        overrides["dtype"] = dtype

    gc = get_bool(f"  {AD} Gradient checkpointing", t.get("gradient_checkpointing", True))
    if gc != t.get("gradient_checkpointing"):
        overrides["gradient_checkpointing"] = gc

    cm = get_bool(f"  {AD} Compile model (torch.compile)", t.get("compile_model", True))
    if cm != t.get("compile_model"):
        overrides["compile_model"] = cm

    fa = get_bool(f"  {AD} Flash attention", t.get("flash_attention", True))
    if fa != t.get("flash_attention"):
        overrides["flash_attention"] = fa

    print()
    use_lora = get_bool(f"  {TW} Use LoRA (add new languages without full retrain)", t.get("use_lora", False))
    if use_lora != t.get("use_lora"):
        overrides["use_lora"] = use_lora
    if use_lora:
        for field, label, getter in [
            ("lora_r", "  LoRA rank (encoder)", lambda: get_int("  LoRA rank encoder", t.get("lora_r", 16), 1)),
            ("lora_r_decoder", "  LoRA rank (decoder)", lambda: get_int("  LoRA rank decoder", t.get("lora_r_decoder", 64), 1)),
        ]:
            val = getter()
            if val != t.get(field):
                overrides[field] = val

    print()
    for field, label, hint, getter in [
        ("save_every", "  Save checkpoint every N epochs", AD, lambda: get_int("  Save every", t.get("save_every", 2), 1)),
        ("validate_every", "  Validate every N epochs", AD, lambda: get_int("  Validate every", t.get("validate_every", 1), 1)),
        ("log_every", "  Log every N steps", AD, lambda: get_int("  Log every", t.get("log_every", 50), 1)),
    ]:
        print(f"  {hint} {label}")
        val = getter()
        if val != t.get(field):
            overrides[field] = val

    return {"training": overrides} if overrides else {}


# ── Section 3: Model Architecture ───────────────────────────────────

def section_model(config: dict) -> dict:
    _section_header(3, 9, "Model Architecture",
                    f"{AD} = only change if you know what you're doing")
    m = config.get("model", {})
    overrides = {}

    for field, label, default, getter in [
        ("hidden_dim", "  Hidden dimension", 512, lambda d: get_int("  Hidden dim", d, 64, 2048)),
        ("num_layers", "  Encoder layers", 6, lambda d: get_int("  Encoder layers", d, 1, 48)),
        ("num_heads", "  Encoder heads", 8, lambda d: get_int("  Encoder heads", d, 1, 64)),
        ("decoder_dim", "  Decoder dimension", 512, lambda d: get_int("  Decoder dim", d, 64, 2048)),
        ("decoder_layers", "  Decoder layers", 8, lambda d: get_int("  Decoder layers", d, 1, 48)),
        ("decoder_heads", "  Decoder heads", 8, lambda d: get_int("  Decoder heads", d, 1, 64)),
    ]:
        print(f"  {AD} {label}")
        val = getter(m.get(field, default))
        if val != m.get(field, default):
            overrides[field] = val

    print(f"  {AD} Dropout")
    dropout = get_float("  Dropout", m.get("dropout", 0.1))
    if abs(dropout - m.get("dropout", 0.1)) > 1e-10:
        overrides["dropout"] = dropout

    return {"model": overrides} if overrides else {}


# ── Section 4: Data ─────────────────────────────────────────────────

def section_data(config: dict) -> dict:
    _section_header(4, 9, "Data Settings",
                    f"{TW} = commonly changed  |  {AD} = advanced")
    d = config.get("data", {})
    overrides = {}

    print(f"  {TW} Languages")
    langs = d.get("languages", [])
    new_langs = get_str_list("  Languages (comma-sep, e.g. en,es,fr,de)", langs)
    if new_langs != langs:
        overrides["languages"] = new_langs

    print(f"  {AD} Random seed")
    seed = get_int("  Seed", d.get("seed", 42), 0)
    if seed != d.get("seed"):
        overrides["seed"] = seed

    print(f"  {AD} Max sentence length (tokens)")
    max_len = get_int("  Max length", d.get("max_sentence_length", 128), 16, 512)
    if max_len != d.get("max_sentence_length"):
        overrides["max_sentence_length"] = max_len

    print(f"\n  {TW} Per-pair sentence counts (training_distribution)")
    td = d.get("training_distribution", {})
    if get_bool("  Adjust per-pair counts?", False):
        new_td = {}
        for pair in sorted(td.keys()):
            cur = td[pair]
            val = get_input(f"    {pair}", str(cur))
            if val:
                try:
                    n = int(val)
                    if n > 0: new_td[pair] = n
                except ValueError:
                    new_td[pair] = cur
        if new_td != td:
            overrides["training_distribution"] = new_td

    print(f"\n  {AD} Augmentation pairs (for backtranslation)")
    ap = d.get("augmentation_pairs", [])
    new_ap = get_str_list("  Augmentation pairs (comma-sep)", ap)
    if new_ap != ap:
        overrides["augmentation_pairs"] = new_ap

    print(f"\n  {AD} Download settings")
    dw = get_int("  Max download workers", d.get("download_max_workers", 4), 1, 32)
    if dw != d.get("download_max_workers"):
        overrides["download_max_workers"] = dw
    pb = get_bool("  Parallel batch downloads", d.get("download_parallel_batches", False))
    if pb != d.get("download_parallel_batches"):
        overrides["download_parallel_batches"] = pb

    print(f"\n  {AD} Vocabulary strategy")
    vs = d.get("vocabulary_strategy", {})
    approach = get_input("  Vocab approach (production/research)", vs.get("approach", "production"))
    if approach.lower() in ("production", "research") and approach.lower() != vs.get("approach"):
        overrides.setdefault("vocabulary_strategy", {})["approach"] = approach.lower()

    return {"data": overrides} if overrides else {}


# ── Section 5: Data Strategy ────────────────────────────────────────

def section_data_strategy(config: dict) -> dict:
    _section_header(5, 9, "Data Strategy",
                    f"{AD} = only change if customizing data sources")
    ds = config.get("data_strategy", {})
    overrides = {}

    pr = ds.get("priority_rules", {})
    print(f"  Current high priority: {len(pr.get('high', []))} pairs")
    print(f"  Current medium priority: {len(pr.get('medium', []))} pairs")
    print(f"  Current low priority: {len(pr.get('low', []))} pairs")
    if get_bool("  Edit priority rules?", False):
        pr_override = {}
        for tier in ("high", "medium", "low"):
            cur = pr.get(tier, [])
            new_list = get_str_list(f"    {tier} pairs", cur)
            if new_list != cur:
                pr_override[tier] = new_list
        if pr_override:
            overrides["priority_rules"] = pr_override

    print(f"\n  Source preferences control which datasets are used per language group.")
    sp = ds.get("source_preferences", {})
    if get_bool("  Edit source preferences?", False):
        sp_override = {}
        for group, sources in sp.items():
            new_str = get_str_list(f"    {group} sources", sources)
            if new_str != sources:
                sp_override[group] = new_str
        if sp_override:
            overrides["source_preferences"] = sp_override

    return {"data_strategy": overrides} if overrides else {}


# ── Section 6: Pipeline Settings ────────────────────────────────────

def section_pipeline(config: dict) -> dict:
    _section_header(6, 9, "Pipeline Settings",
                    f"{TW} = commonly tweaked  |  {AD} = advanced")
    p = config.get("pipeline", {})
    overrides = {}

    print(f"  {TW} COMET quality threshold (0.0-1.0, higher = stricter)")
    ct = get_float("  Threshold", p.get("comet_quality_threshold", 0.7))
    if abs(ct - p.get("comet_quality_threshold", 0.7)) > 1e-10 and 0 <= ct <= 1:
        overrides["comet_quality_threshold"] = ct

    print(f"  {TW} High-resource threshold (skip NLLB BT for pairs above this count)")
    hr = get_int("  Threshold", p.get("high_resource_threshold", 100_000_000), 0)
    if hr != p.get("high_resource_threshold"):
        overrides["high_resource_threshold"] = hr

    print(f"  {AD} Max false friends per pair")
    ff = get_int("  Max FF/pair", p.get("max_dynamic_ff_per_pair", 25000), 0)
    if ff != p.get("max_dynamic_ff_per_pair"):
        overrides["max_dynamic_ff_per_pair"] = ff

    print(f"  {AD} Max idioms per language")
    im = get_int("  Max idioms/lang", p.get("max_idiom_per_lang", 10000), 0)
    if im != p.get("max_idiom_per_lang"):
        overrides["max_idiom_per_lang"] = im

    return {"pipeline": overrides} if overrides else {}


# ── Section 7: Distributed Training ─────────────────────────────────

def section_distributed(config: dict) -> dict:
    _section_header(7, 9, "Distributed Training",
                    f"{AD} = only needed for multi-GPU training")
    di = config.get("distributed", {})
    overrides = {}

    backend = get_input(f"  {AD} Backend (nccl/gloo/mpi)", di.get("backend", "nccl"))
    if backend != di.get("backend"):
        overrides["backend"] = backend

    bucket = get_int(f"  {AD} Bucket cap (MB)", di.get("bucket_cap_mb", 25), 1)
    if bucket != di.get("bucket_cap_mb"):
        overrides["bucket_cap_mb"] = bucket

    for field, label, default in [
        ("find_unused_parameters", "  Find unused parameters", False),
        ("broadcast_buffers", "  Broadcast buffers", False),
        ("gradient_as_bucket_view", "  Gradient as bucket view", True),
        ("static_graph", "  Static graph", True),
    ]:
        val = get_bool(f"  {AD} {label}", di.get(field, default))
        if val != di.get(field, default):
            overrides[field] = val

    return {"distributed": overrides} if overrides else {}


# ── Section 8: Monitoring ───────────────────────────────────────────

def section_monitoring(config: dict) -> dict:
    _section_header(8, 9, "Monitoring & Logging",
                    f"{OP} = optional, enable only if you use the tool")
    mo = config.get("monitoring", {})
    overrides = {}

    for field, label in [
        ("use_wandb", "Use Weights & Biases"),
        ("use_tensorboard", "Use TensorBoard"),
        ("log_gradients", "Log gradient histograms"),
        ("log_weights", "Log weight histograms"),
        ("log_learning_rate", "Log learning rate"),
    ]:
        val = get_bool(f"  {OP} {label}", mo.get(field, False))
        if val != mo.get(field, False):
            overrides[field] = val

    return {"monitoring": overrides} if overrides else {}


# ── Section 9: Hub Settings ─────────────────────────────────────────

def section_hub(config: dict) -> dict:
    _section_header(9, 9, "Hugging Face Hub Settings",
                    f"{TW} = set these if using HF Hub sync")
    h = config.get("hub", {})
    overrides = {}

    dr = get_input(f"  {TW} Dataset repo ID", h.get("dataset_repo_id", "code-with-zeeshan/UTS-Datasets"))
    if dr != h.get("dataset_repo_id"):
        overrides["dataset_repo_id"] = dr

    mr = get_input(f"  {TW} Model repo ID", h.get("model_repo_id", "code-with-zeeshan/Universal-Translation-System"))
    if mr != h.get("model_repo_id"):
        overrides["model_repo_id"] = mr

    print(f"  {OP} Auto-upload after pipeline/training")
    au = get_bool("  Auto-upload", h.get("auto_upload", False))
    if au != h.get("auto_upload"):
        overrides["auto_upload"] = au

    print(f"  {OP} Auto-download data if missing")
    ad = get_bool("  Auto-download", h.get("auto_download", True))
    if ad != h.get("auto_download"):
        overrides["auto_download"] = ad

    return {"hub": overrides} if overrides else {}


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
    print(f"    \033[1;37m$ uts eval --model --config {path}\033[0m")
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive config builder for Universal Translation System",
        epilog="Examples:\n  python scripts/config_interactive.py\n  python scripts/config_interactive.py --set training.num_epochs=20",
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
    parser.add_argument("--preset-stages", default="",
                        help="Comma-separated stages to pre-fill (from data --interactive)")
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

    preset_stages = [s.strip() for s in args.preset_stages.split(",") if s.strip()] if args.preset_stages else None

    clear_screen()
    print("\033[1;36m\u2500\u2500 Universal Translation System \u2014 Interactive Config Builder \u2500\u2500\033[0m\n")
    print("  This wizard helps you create a config based on base.yaml.")
    print("  Each setting shows its default; press ENTER to keep it.\n")
    print(f"  Legend:  {TW} = commonly changed  |  {AD} = power user  |  {OP} = optional\n")
    print("  Sections: 1) Pipeline Stages  2) Training  3) Model  4) Data")
    print("            5) Data Strategy  6) Pipeline  7) Distributed")
    print("            8) Monitoring  9) Hub  10) Review & Save")
    wait_enter()

    sections = [
        ("Pipeline Stages", lambda: section_stages(base, preset_stages)),
        ("Training", lambda: section_training(merged)),
        ("Model Architecture", lambda: section_model(merged)),
        ("Data Settings", lambda: section_data(merged)),
        ("Data Strategy", lambda: section_data_strategy(merged)),
        ("Pipeline Settings", lambda: section_pipeline(merged)),
        ("Distributed Training", lambda: section_distributed(merged)),
        ("Monitoring & Logging", lambda: section_monitoring(merged)),
        ("Hub Settings", lambda: section_hub(merged)),
    ]

    for name, fn in sections:
        result = fn()
        if result is None:
            print("  \033[33mCancelled.\033[0m")
            return 1
        merged = deep_merge(merged, result)
        wait_enter()

    result = section_review(base, merged)
    if result is None:
        print("  \033[33mCancelled.\033[0m")
        return 1
    print(f"\n  \033[1;32mDone!\033[0m Configuration saved to \033[1;33m{result}\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
