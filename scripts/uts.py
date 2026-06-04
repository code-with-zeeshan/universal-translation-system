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
  uts serve          Start decoder / coordinator services
  uts tools          Utilities (config, GPU check, secrets, etc.)
  uts docs           Open documentation

  uts <group> --help    Detailed help for a group
  uts help <topic>      Open documentation in browser
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


# ── helpers ──────────────────────────────────────────────────────────

def _run(*args: str, **kwargs):
    """Run a python module as `python -m module [args...]`."""
    cmd = [PY, "-m"] + list(args)
    if kwargs:
        for k, v in kwargs.items():
            if v is not None:
                cmd.append(f"--{k.replace('_', '-')}")
                if not isinstance(v, bool):
                    cmd.append(str(v))
    print(f"\n$ {' '.join(shlex.join(c) for c in cmd)}\n")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_script(script: str, *args: str):
    """Run a script under scripts/."""
    cmd = [PY, str(ROOT / script)] + list(args)
    print(f"\n$ {' '.join(shlex.join(c) for c in cmd)}\n")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=True)


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
    parser.print_help()


def build_setup_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--check", action="store_true", help="Run GPU / environment readiness check")
    sub.add_argument("--config-wizard", action="store_true", help="Interactive config wizard")
    sub.add_argument("--validate", metavar="CONFIG_PATH", help="Validate a config YAML file")


# ── data ─────────────────────────────────────────────────────────────

def cmd_data(args: argparse.Namespace):
    if args.pipeline:
        _run("data.unified_data_pipeline", config=args.config, resume=args.resume,
             stage=args.stage, reset=args.reset)
    elif args.download_only:
        _run("data.unified_data_pipeline", config=args.config, eval_only=True)
    elif args.augment:
        _run_module("data/synthetic_augmentation.py")
    elif args.validate_data:
        _run("data.unified_data_pipeline", config=args.config, stage="validate")
    else:
        print("See: uts data --help")


def build_data_parser(sub: argparse.ArgumentParser):
    sub.add_argument("--pipeline", action="store_true", help="Run the full data pipeline")
    sub.add_argument("--download-only", action="store_true", help="Download evaluation data only")
    sub.add_argument("--augment", action="store_true", help="Run synthetic data augmentation")
    sub.add_argument("--validate-data", action="store_true", help="Validate pipeline output")
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    sub.add_argument("--reset", action="store_true", help="Reset pipeline state (start fresh)")
    sub.add_argument("--stage", help="Run a single pipeline stage")


# ── vocab ────────────────────────────────────────────────────────────

def cmd_vocab(args: argparse.Namespace):
    if args.build:
        _run_script("scripts/build_and_upload_pipeline.py",
                     "--create-vocabs",
                     *(("--vocab-size", str(args.vocab_size)) if args.vocab_size else []),
                     *(("--vocab-mode", args.mode) if args.mode else []),
                     *(("--vocab-groups", *args.groups) if args.groups else []))
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

def cmd_train(args: argparse.Namespace):
    if args.full:
        _run("training.trainer" if args.distributed else "training.launch",
             "train",
             config=args.config,
             distributed=args.distributed,
             num_epochs=args.num_epochs,
             batch_size=args.batch_size,
             learning_rate=args.lr,
             experiment_name=args.experiment_name,
             checkpoint=args.checkpoint)
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
    sub.add_argument("--progressive", action="store_true", help="Progressive multi-tier training")
    sub.add_argument("--lora", action="store_true", help="Show LoRA adapter training instructions")
    sub.add_argument("--config", default="config/base.yaml", help="Config file path")
    sub.add_argument("--distributed", action="store_true", help="Enable distributed training")
    sub.add_argument("--num-epochs", type=int, help="Override number of epochs")
    sub.add_argument("--batch-size", type=int, help="Override batch size")
    sub.add_argument("--lr", type=float, help="Override learning rate")
    sub.add_argument("--experiment-name", help="Experiment name for logging")
    sub.add_argument("--checkpoint", help="Resume from checkpoint path")
    sub.add_argument("--start-tier", choices=["tier1", "tier2", "tier3", "tier4"],
                     help="Progressive: start from a specific tier")
    sub.add_argument("--validate-final", action="store_true",
                     help="Progressive: validate final model")


# ── eval ─────────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace):
    if args.model:
        _run_module("evaluation/evaluate_model.py",
                     config=args.config,
                     checkpoint=args.checkpoint,
                     test_data=args.test_data)
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
        _run_script("scripts/upload_artifacts.py", f"--repo_id={args.repo_id or args.upload}")
    elif args.version:
        _run_script("scripts/version_manager.py", "show")
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
    "roadmap": "docs/Roadmap.md",
    "vision": "docs/VISION.md",
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
  serve          Services: decoder server, coordinator, Redis
  tools          Utilities: config, GPU, secrets, upload, prefetch
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
  uts help train                          Open training docs
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

# serve
p_serve = subparsers.add_parser("serve", help="Start serving infrastructure")
build_serve_parser(p_serve)
p_serve.set_defaults(func=cmd_serve)

# tools
p_tools = subparsers.add_parser("tools", help="Utility tools")
build_tools_parser(p_tools)
p_tools.set_defaults(func=cmd_tools)

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
        print()
        print("  uts docs --open setup     Full setup guide")
        print("  uts docs --open train     Training guide")
        return
    args.func(args)


if __name__ == "__main__":
    main()
