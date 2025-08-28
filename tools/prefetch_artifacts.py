#!/usr/bin/env python3
"""
Prefetch models, vocabulary packs, and adapters from Hugging Face Hub to local dirs.

Examples:
  python tools/prefetch_artifacts.py --pairs en:es en:fr --adapters es fr --models production/encoder.onnx
  python tools/prefetch_artifacts.py --packs latin cjk --version 1.0

Env needed:
- HF_HUB_REPO_ID (or use --repo_id)
- HF_TOKEN if repo is private
Optional overrides: HF_HUB_REVISION, MODELS_DIR, VOCABS_DIR, ADAPTERS_DIR
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os
import sys
import logging

from utils.artifact_store import ArtifactStore, StoreConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Prefetch artifacts from HF Hub")
    p.add_argument("--repo_id", help="HF repo id (default from HF_HUB_REPO_ID)")
    p.add_argument("--revision", default=None, help="HF revision (branch/tag/sha), default env or 'main'")
    p.add_argument("--pairs", nargs='*', help="Language pairs like en:es en:fr")
    p.add_argument("--packs", nargs='*', help="Vocabulary pack names like latin cjk arabic")
    p.add_argument("--version", help="Explicit pack version (e.g., 1.0)")
    p.add_argument("--adapters", nargs='*', help="Adapter names to fetch (e.g., es fr ja)")
    p.add_argument("--models", nargs='*', help="Repo paths under models/, e.g., production/encoder.onnx")
    return p.parse_args()


def make_store(repo_id: Optional[str], revision: Optional[str]) -> ArtifactStore:
    # Build config from env + args
    env_repo = os.environ.get("HF_HUB_REPO_ID")
    rid = repo_id or env_repo
    if not rid:
        logger.error("HF_HUB_REPO_ID not set and --repo_id not provided")
        sys.exit(2)
    token = os.environ.get("HF_TOKEN")
    rev = revision or os.environ.get("HF_HUB_REVISION", "main")
    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    vocabs_dir = Path(os.environ.get("VOCABS_DIR", "vocabs"))
    adapters_dir = Path(os.environ.get("ADAPTERS_DIR", str(models_dir / "adapters")))

    cfg = StoreConfig(
        repo_id=rid, token=token, revision=rev,
        models_dir=models_dir, vocabs_dir=vocabs_dir, adapters_dir=adapters_dir
    )
    return ArtifactStore(cfg)


def main():
    args = parse_args()
    store = make_store(args.repo_id, args.revision)

    # Prefetch models
    if args.models:
        for m in args.models:
            try:
                store.ensure_model(m)
            except Exception as e:
                logger.error(f"Failed to fetch model {m}: {e}")

    # Prefetch vocab packs
    if args.packs:
        for pack in args.packs:
            try:
                store.ensure_vocab_pack(pack, version=args.version)
            except Exception as e:
                logger.error(f"Failed to fetch vocab pack {pack}: {e}")

    # Prefetch adapters
    if args.adapters:
        for a in args.adapters:
            try:
                store.ensure_adapter(a)
            except Exception as e:
                logger.error(f"Failed to fetch adapter {a}: {e}")

    # Prefetch by pairs
    if args.pairs:
        for pair in args.pairs:
            try:
                src, tgt = pair.split(":", 1)
            except ValueError:
                logger.error(f"Invalid pair '{pair}', expected format src:tgt")
                continue
            try:
                store.ensure_for_language_pair(src, tgt)
            except Exception as e:
                logger.error(f"Failed to ensure artifacts for {src}:{tgt}: {e}")


if __name__ == "__main__":
    main()