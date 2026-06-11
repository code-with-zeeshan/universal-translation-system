#!/usr/bin/env python3
"""
End-to-end pipeline to:
1) Build/convert final models as needed
2) Create vocabulary packs
3) Upload models, adapters, and vocabs to Hugging Face sequentially

Usage:
  python scripts/build_and_upload_pipeline.py \
      --repo_id code-with-zeeshan/universal-translation-system \
      --create-vocabs --convert-models

Environment requirements:
- HF_TOKEN must be set for pushing to HF if the repo is private or for auth
- Optional: HF_HUB_REVISION (default: main)

This script reuses:
- training/convert_models.py for conversion
- vocabulary/unified_vocabulary_creator.py for pack creation
- scripts/upload_artifacts.py for upload
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def run_py(module_path: str, args: list[str] | None = None):
    cmd = [sys.executable, module_path]
    if args:
        cmd.extend(args)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dirs():
    for p in [ROOT / "models/production", ROOT / "models/adapters", ROOT / "vocabulary/vocab"]:
        p.mkdir(parents=True, exist_ok=True)


def create_vocabs(groups: list[str] | None = None, mode: str | None = None, vocab_size: int = 32000):
    # Run vocabulary creator via a small Python call to avoid import context issues
    code = (
        "from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator, CreationMode;"
        "from vocabulary.vocab_config import UnifiedVocabConfig;"
        "creator=UnifiedVocabularyCreator(corpus_dir='data/processed', output_dir='vocabulary/vocab',"
        "config=UnifiedVocabConfig(vocab_size=%d));" % vocab_size
        + ("mode=CreationMode.%s;" % mode.upper() if mode else "mode=None;")
        + ("groups=%s;" % groups if groups else "groups=None;")
        + "creator.create_all_packs(mode=mode, groups_to_create=groups);"
    )
    logger.info("Creating vocabulary packs%s%s", f" (mode={mode})" if mode else "", f" groups={groups}" if groups else "")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(ROOT))


def convert_models():
    # Use existing conversion pipeline
    run_py(str(ROOT / "training/convert_models.py"))


def upload(repo_id: str):
    # Reuse existing upload script
    run_py(str(ROOT / "scripts/upload_artifacts.py"), ["--repo-id", repo_id])


def main():
    parser = argparse.ArgumentParser(description="Build and upload artifacts pipeline")
    parser.add_argument("--repo-id", required=True, dest="repo_id", help="HF Hub repo id, e.g. code-with-zeeshan/universal-translation-system")
    parser.add_argument("--create-vocabs", action="store_true", help="Create vocabulary packs before upload")
    parser.add_argument("--convert-models", action="store_true", help="Run model conversion before upload")
    parser.add_argument("--vocab-groups", nargs='*', help="Specific vocab groups to build (latin cjk arabic devanagari cyrillic thai)")
    parser.add_argument("--vocab-mode", choices=["production", "research", "hybrid"], help="Vocabulary creation mode override")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokens per vocabulary pack (default: 32000)")
    args = parser.parse_args()

    ensure_dirs()

    # Optional steps
    if args.create_vocabs:
        create_vocabs(groups=args.vocab_groups, mode=args.vocab_mode, vocab_size=args.vocab_size)

    if args.convert_models:
        convert_models()

    # Upload sequentially at the end
    upload(args.repo_id)


if __name__ == "__main__":
    # Validate HF token if needed
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set. If your repo requires auth, uploads may fail. Consider `huggingface-cli login`.")
    main()