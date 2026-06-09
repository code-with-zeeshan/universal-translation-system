#!/usr/bin/env python3
"""
uts — Universal Translation System unified CLI.

Single entry point that organizes all tools by workflow.
Run without arguments for an overview, or use subcommands.

  uts setup          Environment setup and validation
  uts data           Data pipeline (download, augment, validate)
  uts vocab          Vocabulary pack management
  uts train          Model training (full / progressive / LoRA)
  uts eval           Evaluation and benchmarking
  uts publish        Publish trained model to Hugging Face Hub
  uts serve          Start decoder / coordinator services
  uts tools          Utilities (config, GPU check, secrets, etc.)
  uts docs           Open documentation

  uts <group> --help    Detailed help for a group
  uts docs --open <topic>      Open documentation in browser
"""

import sys
import os
from pathlib import Path
import subprocess
import shlex
import json
import yaml
import argparse
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from utils.pipeline_checkpoint import mark_stage_complete, hash_config

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


# ── helpers ──────────────────────────────────────────────────────────

def _scale_config(config_path: str, scale: float) -> str:
    """Read a YAML config, scale training_distribution values, write to temp file."""
    try:
        import yaml
    except ImportError:
        print("error: PyYAML required for --scale. Run: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    td = cfg.get("data", {}).get("training_distribution", {})
    if not td:
        print("warning: training_distribution not found in config")
        return config_path

    for k in td:
        td[k] = int(td[k] * scale)

    # Also scale total_size_gb proportionally
    cfg.setdefault("data", {})["total_size_gb"] = max(1, int(cfg["data"].get("total_size_gb", 2) * scale))

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    print(f"→ Scaled training_distribution by {scale}× → {tmp.name}")
    return tmp.name


def _find_latest_checkpoint() -> str | None:
    """Return the most recently modified .pt file in checkpoints/."""
    candidates = sorted(Path("checkpoints").rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        print(f"→ Auto-detected checkpoint: {candidates[0]}")
        return str(candidates[0])
    return None


def _run(*args: str, **kwargs):
    """Run a python module as `python -m module [args...]`."""
    cmd = [PY, "-m"] + list(args)
    if kwargs:
        for k, v in kwargs.items():
            if v is not None and v is not False:
                cmd.append(f"--{k.replace('_', '-')}")
                if v is not True:
                    cmd.append(str(v))
    print(f"\n$ {' '.join(shlex.join(c) for c in cmd)}\n")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_script(script: str, *args: str, check: bool = True):
    """Run a script under scripts/."""
    cmd = [PY, str(ROOT / script)] + list(args)
    print(f"\n$ {' '.join(shlex.join(c) for c in cmd)}\n")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=check)


def _run_module(module: str, *args: str):
    """Run `python module_path.py [args...]`."""
    cmd = [PY, str(ROOT / module)] + list(args)
    print(f"\n$ {' '.join(shlex.join(c) for c in cmd)}\n")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=True)


# ── setup ────────────────────────────────────────────────────────────

def cmd_setup(args: argparse.Namespace):
    if args.check:
        _run("scripts.gpu_readiness_check")
        return
    if args.config_wizard:
        _run_script("scripts/config_wizard.py")
        return
    if args.validate:
        _run_script("scripts/validate_config.py", args.validate)
        return
    if args.verify:
        _run_script("scripts/first_time_success.py", check=False)
        return
    parser.print_help()


def build_setup_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--check", action="store_true", help="Run GPU / environment readiness check")
    sub.add_argument("--config-wizard", action="store_true", help="Interactive config wizard")
    sub.add_argument("--validate", metavar="CONFIG_PATH", help="Validate a config YAML file")
    sub.add_argument("--verify", action="store_true", help="Verify post-deployment setup (check services, endpoints, sample translation)")


# ── data ─────────────────────────────────────────────────────────────

def cmd_data(args: argparse.Namespace):
    config_path = args.config
    if args.scale and args.scale != 1.0:
        config_path = _scale_config(config_path, args.scale)
    if args.pipeline:
        _run("data.unified_data_pipeline", config=config_path, resume=args.resume,
             force=args.force if hasattr(args, 'force') else False,
             stage=args.stage, reset=args.reset,
             download_max_workers=args.download_max_workers,
             download_parallel_batches=args.download_parallel_batches,
             datasets_cache_dir=args.datasets_cache_dir)
        # Mark data pipeline complete in global state
        try:
            with open(config_path) as f:
                raw = json.load(f)
            dh = hash_config(raw.get("data", {}))
            mark_stage_complete("data", dh)
        except Exception:
            pass
    elif args.download_only:
        _run("data.unified_data_pipeline", config=config_path, eval_only=True,
             download_max_workers=args.download_max_workers,
             datasets_cache_dir=args.datasets_cache_dir)
    elif args.augment:
        _run_module("data/synthetic_augmentation.py")
    elif args.validate_data:
        _run("data.unified_data_pipeline", config=config, stage="validate")
    elif args.domains:
        _run_module("data/acquire_domain_data.py", "--domains", args.domains)
    else:
        print("See: uts data --help")


def build_data_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--pipeline", action="store_true", help="Run the full data pipeline")
    sub.add_argument("--download-only", action="store_true", help="Download evaluation data only")
    sub.add_argument("--augment", action="store_true", help="Run synthetic data augmentation")
    sub.add_argument("--validate-data", action="store_true", help="Validate pipeline output")
    sub.add_argument("--domains", metavar="LIST", help="Download domain-specific data (e.g., medical,legal)")
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--scale", type=float, default=1.0,
                     help="Scale training_distribution by factor (e.g., --scale 5 for 5× data)")
    sub.add_argument("--resume", action="store_true", default=True,
                     help="Resume from last checkpoint (default: auto-detect)")
    sub.add_argument("--no-resume", action="store_false", dest="resume",
                     help="Ignore checkpoint and run all stages")
    sub.add_argument("--force", action="store_true",
                     help="Clear checkpoint and re-run all stages from scratch")
    sub.add_argument("--reset", action="store_true", help="Reset pipeline state (start fresh)")
    sub.add_argument("--stage", help="Run a single pipeline stage")
    sub.add_argument("--download-max-workers", type=int, default=None,
                     help="Override max parallel downloads per batch (default: 4)")
    sub.add_argument("--download-parallel-batches", action="store_true",
                     help="Enable parallel batch downloads")
    sub.add_argument("--datasets-cache-dir", type=str, default=None,
                     help="HuggingFace datasets cache directory (default: HF default cache)")


# ── vocab ────────────────────────────────────────────────────────────

def cmd_vocab(args: argparse.Namespace):
    if args.build:
        code = (
            "from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator, CreationMode;"
            "from vocabulary.vocab_config import UnifiedVocabConfig;"
            "creator=UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabulary/vocab',"
            "config=UnifiedVocabConfig(vocab_size=%d));" % (args.vocab_size or 32000)
            + ("mode=CreationMode.%s;" % args.mode.upper() if args.mode else "mode=None;")
            + ("groups=%s;" % args.groups if args.groups else "groups=None;")
            + "creator.create_all_packs(mode=mode, groups_to_create=groups);"
        )
        print(f"\n$ python -c \"vocabulary creator\"\n")
        sys.stdout.flush()
        subprocess.run([PY, "-c", code], check=True, cwd=str(ROOT))
    elif args.evolve:
        _run_module("vocabulary/evolve_vocabulary.py",
                     *(("--pack", args.pack) if args.pack else []))
    else:
        print("See: uts vocab --help")


def build_vocab_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--build", action="store_true", help="Build vocabulary packs from processed data")
    sub.add_argument("--evolve", action="store_true", help="Evolve an existing vocabulary pack")
    sub.add_argument("--vocab-size", type=int, default=32000, help="Tokens per vocabulary pack")
    sub.add_argument("--mode", choices=["production", "research", "hybrid"], help="Creation mode")
    sub.add_argument("--groups", nargs="*", help="Specific groups: latin cjk arabic devanagari cyrillic thai")
    sub.add_argument("--pack", help="Specific pack to evolve")


# ── train ────────────────────────────────────────────────────────────

def _data_config_hash(config_path: str) -> str:
    """Fingerprint the data pipeline config."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        data_section = cfg.get("data", {})
        return hash_config(data_section)
    except Exception:
        return ""


def cmd_train(args: argparse.Namespace):
    if args.full:
        checkpoint = args.checkpoint or _find_latest_checkpoint()
        _run("training.trainer" if args.distributed else "training.launch",
             "train",
             config=args.config,
             distributed=args.distributed,
             num_epochs=args.num_epochs,
             batch_size=args.batch_size,
             learning_rate=args.lr,
             experiment_name=args.experiment_name,
             checkpoint=checkpoint,
             force=args.force if hasattr(args, 'force') else False)
    elif args.distill:
        _run_module("training/distillation_trainer.py",
                     config=args.config,
                     teacher=args.teacher or "facebook/nllb-200-3.3B",
                     alpha=args.distill_alpha,
                     temperature=args.distill_temp,
                     num_epochs=args.num_epochs)
    elif args.progressive:
        _run_module("training/progressive_training.py",
                     *(("--start-from-tier", args.start_tier) if args.start_tier else []),
                     *(("--validate-final",) if args.validate_final else []))
    elif args.lora:
        print("NOTE: For LoRA adapter training (adding new languages after full training),")
        print("      set use_lora: true in config/base.yaml and run:")
        print("      uts train --full")
    else:
        print("See: uts train --help")


def build_train_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--full", action="store_true", help="Full model training (all parameters)")
    sub.add_argument("--distill", action="store_true", help="Knowledge distillation from a teacher model")
    sub.add_argument("--teacher", default="facebook/nllb-200-3.3B", help="Teacher model for distillation")
    sub.add_argument("--distill-alpha", type=float, default=0.5, help="CE vs KD loss weight (0-1)")
    sub.add_argument("--distill-temp", type=float, default=4.0, help="Distillation temperature")
    sub.add_argument("--progressive", action="store_true", help="Progressive multi-tier training")
    sub.add_argument("--lora", action="store_true", help="Show LoRA adapter training instructions")
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--distributed", action="store_true", help="Enable distributed training")
    sub.add_argument("--num-epochs", type=int, help="Override number of epochs")
    sub.add_argument("--batch-size", type=int, help="Override batch size")
    sub.add_argument("--lr", type=float, help="Override learning rate")
    sub.add_argument("--experiment-name", help="Experiment name for logging")
    sub.add_argument("--checkpoint", help="Resume from checkpoint path")
    sub.add_argument("--force", action="store_true",
                     help="Ignore training checkpoint and re-train from scratch")
    sub.add_argument("--start-tier", choices=["tier1", "tier2", "tier3", "tier4"],
                     help="Progressive: start from a specific tier")
    sub.add_argument("--validate-final", action="store_true",
                     help="Progressive: validate final model")


# ── eval ─────────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace):
    if args.model:
        eval_dir = Path(args.test_data or "data/evaluation")
        if not eval_dir.exists() or not any(eval_dir.iterdir()):
            print("→ Eval data missing, downloading...")
            _run("data.unified_data_pipeline", config=args.config, eval_only=True)
        eval_args = [f"--config={args.config}",
                     f"--checkpoint={args.checkpoint}",
                     f"--test-data={args.test_data}"]
        if hasattr(args, 'force') and args.force:
            eval_args.append("--force")
        _run_module("evaluation/evaluate_model.py", *eval_args)
    elif args.download:
        _run("data.unified_data_pipeline", config=args.config, eval_only=True)
    elif args.benchmark:
        _run("training.launch", "profile",
             config=args.config,
             profile_steps=args.profile_steps,
             benchmark=True,
             output_dir=args.output_dir)
    else:
        print("See: uts eval --help")


def build_eval_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--model", action="store_true", help="Evaluate a trained model checkpoint")
    sub.add_argument("--download", action="store_true", help="Download evaluation test data")
    sub.add_argument("--benchmark", action="store_true", help="Benchmark model performance")
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--checkpoint", help="Path to model checkpoint (.pt)")
    sub.add_argument("--test-data", default="data/evaluation", help="Test data directory")
    sub.add_argument("--profile-steps", type=int, default=10, help="Steps to profile")
    sub.add_argument("--output-dir", default="profiling", help="Profiling output directory")
    sub.add_argument("--force", action="store_true",
                     help="Re-evaluate all files even if previously completed")


# ── publish ─────────────────────────────────────────────────────────

def cmd_publish(args: argparse.Namespace):
    cmd = ["--repo-id", args.repo_id]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.no_onnx:
        cmd += ["--no-onnx"]
    if args.no_quantize:
        cmd += ["--no-quantize"]
    if args.upload_only:
        cmd += ["--upload-only"]
    if args.preflight:
        _run_module("tools/cloud_preflight.py")
        return
    if args.optimize_decoder:
        _run_module("cloud_decoder/quantize_optimize.py")
        return
    _run_script("scripts/publish.py", *cmd)


def build_publish_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--repo-id", required=True, help="HF Hub repo ID (e.g., your-org/universal-translation-system)")
    sub.add_argument("--checkpoint", help="Path to best_model.pt (auto-discovers)")
    sub.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    sub.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    sub.add_argument("--upload-only", action="store_true", help="Just upload existing artifacts")
    sub.add_argument("--preflight", action="store_true", help="Run cloud preflight checks before publish")
    sub.add_argument("--optimize-decoder", action="store_true", help="Quantize and optimize the decoder model")


# ── serve ────────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace):
    if args.decoder:
        _run_module("cloud_decoder/decoder_server.py")
    elif args.coordinator:
        _run_module("coordinator/advanced_coordinator.py")
    elif args.setup:
        _run_script("scripts/setup_serving.sh", *(["--all"] if args.all else []))
    elif args.redis:
        _run_script("scripts/setup_redis.sh",
                     "--install" if args.install else "--start" if args.start else "--stop" if args.stop else "--status")
    else:
        print("See: uts serve --help")


def build_serve_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--decoder", action="store_true", help="Start the cloud decoder server")
    sub.add_argument("--coordinator", action="store_true", help="Start the coordinator service")
    sub.add_argument("--setup", action="store_true", help="Configure serving infrastructure")
    sub.add_argument("--all", action="store_true", help="Setup all serving components")
    sub.add_argument("--redis", choices=["install", "start", "stop", "status"],
                     help="Manage Redis for decoder pool coordination")


# ── tools ────────────────────────────────────────────────────────────

def cmd_tools(args: argparse.Namespace):
    if args.validate_config:
        _run_script("scripts/validate_config.py", args.validate_config,
                     *(("--check-references",) if args.check_references else []),
                     *(("--check-consistency",) if args.check_consistency else []),
                     *(("--suggest-improvements",) if args.suggest else []),
                     *(("--verbose",) if args.verbose else []))
    elif args.check_gpu:
        _run("scripts.gpu_readiness_check")
    elif args.prefetch:
        _run_module("tools/prefetch_artifacts.py",
                     *(("--pairs", *args.pairs) if args.pairs else []),
                     *(("--packs", *args.packs) if args.packs else []),
                     repo_id=args.repo_id)
    elif args.rotate_secrets:
        _run_module("tools/rotate_secrets.py",
                     type=args.key_type,
                     *(("--set-env",) if args.set_env else []))
    elif args.upload:
        _run_script("scripts/upload_artifacts.py", f"--repo-id={args.repo_id or args.upload}")
    elif args.version:
        _run_script("scripts/version_manager.py", "show")
    elif args.register_decoder:
        _run_module("tools/register_decoder_node.py")
    elif args.build_encoder:
        _run_script("scripts/build_encoder_core.sh", args.build_target or "--all")
    elif args.check_compat:
        _run_script("scripts/compatibility_checks.py",
                     *(("--version-config", args.version_config) if args.version_config else []))
    elif args.install:
        _run_script("scripts/install.sh",
                     *(["--train"] if args.train else []),
                     *(["--serve"] if args.serve else []),
                     *(["--all"] if args.all else []))
    else:
        print("See: uts tools --help")


def build_tools_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--validate-config", metavar="CONFIG_PATH", help="Validate a config file")
    sub.add_argument("--check-references", action="store_true", help="Check referenced files exist")
    sub.add_argument("--check-consistency", action="store_true", help="Check internal consistency")
    sub.add_argument("--suggest", action="store_true", help="Suggest config improvements")
    sub.add_argument("--verbose", action="store_true", help="Verbose validation output")
    sub.add_argument("--check-gpu", action="store_true", help="Run GPU readiness check")
    sub.add_argument("--prefetch", action="store_true", help="Prefetch artifacts from HF Hub")
    sub.add_argument("--pairs", nargs="*", help="Language pairs for prefetch (en:es en:fr)")
    sub.add_argument("--packs", nargs="*", help="Vocab packs for prefetch")
    sub.add_argument("--repo-id", help="HF Hub repository ID")
    sub.add_argument("--rotate-secrets", action="store_true", help="Rotate JWT secrets")
    sub.add_argument("--key-type", choices=["hs256", "rs256", "all"], default="hs256",
                     help="Key type to rotate")
    sub.add_argument("--set-env", action="store_true", help="Set env vars for preview")
    sub.add_argument("--upload", nargs="?", const=True, help="Upload artifacts to HF Hub")
    sub.add_argument("--version", action="store_true", help="Show component versions")
    sub.add_argument("--register-decoder", action="store_true", help="Register a decoder node with the coordinator pool")
    sub.add_argument("--build-encoder", nargs="?", const="--all",
                     help="Build native C++ encoder core for all platforms")
    sub.add_argument("--build-target", help="Target platform for encoder build (auto: --all)")
    sub.add_argument("--check-compat", action="store_true", help="Run API/schema/version compatibility checks")
    sub.add_argument("--version-config", help="Path to version-config.json for compatibility check")
    sub.add_argument("--install", action="store_true", help="Install system dependencies")
    sub.add_argument("--train", action="store_true", help="Install training deps")
    sub.add_argument("--serve", action="store_true", help="Install serving deps")


# ── docs ─────────────────────────────────────────────────────────────

DOCS = {
    "setup": "SETUP_COMMANDS.md",
    "train": "docs/TRAINING.md",
    "arch": "docs/ARCHITECTURE.md",
    "vocab": "docs/Vocabulary_Guide.md",
    "deploy": "docs/DEPLOYMENT.md",
    "api": "docs/API.md",
    "env": "docs/environment-variables.md",
    "sdk": "docs/SDK_INTEGRATION.md",
    "monitor": "monitoring/README.md",
    "faq": "FAQ.md",
    "trouble": "docs/TROUBLESHOOT.md",
    "vision": "docs/VISION.md",
    "layout": "docs/RUNTIME_LAYOUT.md",
    "version": "docs/VERSION_MANAGEMENT.md",
    "secret": "docs/SECRET_MANAGEMENT.md",
    "tui": "docs/TUI.md",
    "publish": "docs/PUBLISHING.md",
    "test": "docs/TESTING.md",
}

TOPICS = {
    k: Path(ROOT / v) for k, v in DOCS.items()
}


def cmd_docs(args: argparse.Namespace):
    if args.open:
        target = TOPICS.get(args.open)
        if target and target.exists():
            if sys.platform == "linux":
                subprocess.run(["xdg-open", str(target)], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                print(target.read_text())
        else:
            known = ", ".join(sorted(TOPICS))
            print(f"Unknown topic '{args.open}'. Known topics: {known}")
        return
    if args.list:
        print("\nDocumentation topics:")
        for k, path in sorted(TOPICS.items()):
            exists = "✓" if path.exists() else "✗"
            print(f"  {k:<12} {exists} {path.name}")
        print("\n  uts docs --open <topic>    Open a topic")
        return
    print("See: uts docs --help")


def build_docs_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--open", metavar="TOPIC", help="Open documentation for a topic")
    sub.add_argument("--list", action="store_true", help="List available documentation topics")


# ── tui ─────────────────────────────────────────────────────────────

def cmd_tui(args: argparse.Namespace):
    _run("tui.app",
         config=args.config,
         pipeline=args.pipeline,
         train=args.train)


def build_tui_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--pipeline", action="store_true", help="Data pipeline only")
    sub.add_argument("--train", action="store_true", help="Training only")


# ── main parser ──────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Universal Translation System  —  unified CLI              ║
║   uts <command> [--help]  to explore                       ║
╚══════════════════════════════════════════════════════════════╝
"""

GROUP_HELP = """\
Workflows (run `uts <group> --help` for details):

  setup          Environment setup, validation, config wizard
  data           Data pipeline: download, augment, validate
  vocab          Vocabulary pack: build or evolve
  train          Training: full model / progressive / LoRA
  eval           Evaluation: model eval, benchmark, data download
  publish        Publish model to Hugging Face Hub (single source of truth)
  serve          Services: decoder server, coordinator, Redis
  tools          Utilities: config, GPU, secrets, upload, prefetch
  tui            Terminal UI dashboard for pipeline/training
  docs           Open documentation by topic

Common flags (available on most groups):
  --config PATH     Config file (default: config/base.yaml)
"""

parser = argparse.ArgumentParser(
    prog="uts",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=BANNER + "\n" + GROUP_HELP,
    epilog="""
Examples:
  uts setup --check                       Check environment
  uts data --pipeline --config base.yaml  Run data pipeline
  uts train --full --config base.yaml     Train full model
  uts tools --validate-config config.yaml Validate config
  uts docs --open train                   Open training docs
""",
)

subparsers = parser.add_subparsers(dest="group", required=False)

# setup
p_setup = subparsers.add_parser("setup", help="Environment setup and validation")
build_setup_parser(p_setup)
p_setup.set_defaults(func=cmd_setup)

# data
p_data = subparsers.add_parser("data", help="Data pipeline operations")
build_data_parser(p_data)
p_data.set_defaults(func=cmd_data)

# vocab
p_vocab = subparsers.add_parser("vocab", help="Vocabulary pack management")
build_vocab_parser(p_vocab)
p_vocab.set_defaults(func=cmd_vocab)

# train
p_train = subparsers.add_parser("train", help="Model training")
build_train_parser(p_train)
p_train.set_defaults(func=cmd_train)

# eval
p_eval = subparsers.add_parser("eval", help="Evaluation and benchmarking")
build_eval_parser(p_eval)
p_eval.set_defaults(func=cmd_eval)

# publish
p_publish = subparsers.add_parser("publish", help="Publish model to Hugging Face Hub")
build_publish_parser(p_publish)
p_publish.set_defaults(func=cmd_publish)

# serve
p_serve = subparsers.add_parser("serve", help="Start serving infrastructure")
build_serve_parser(p_serve)
p_serve.set_defaults(func=cmd_serve)

# tools
p_tools = subparsers.add_parser("tools", help="Utility tools")
build_tools_parser(p_tools)
p_tools.set_defaults(func=cmd_tools)

# tui
p_tui = subparsers.add_parser("tui", help="Terminal UI dashboard for pipeline/training")
build_tui_parser(p_tui)
p_tui.set_defaults(func=cmd_tui)

# docs
p_docs = subparsers.add_parser("docs", help="Documentation browser")
build_docs_parser(p_docs)
p_docs.set_defaults(func=cmd_docs)


def main():
    args = parser.parse_args()
    if not args.group:
        print(BANNER)
        print(GROUP_HELP)
        print("Quick start on Lightning AI:\n")
        print("  1. uts setup --config-wizard")
        print("  2. uts data --pipeline")
        print("  3. uts train --full")
        print("  4. uts eval --model --checkpoint checkpoints/*/best_model.pt")
        print("  5. uts publish --repo-id your-org/universal-translation-system")
        print("  Dashboard: uts tui --config config/base.yaml")
        print()
        print("  uts docs --open setup     Full setup guide")
        print("  uts docs --open train     Training guide")
        return
    args.func(args)


if __name__ == "__main__":
    main()
