#!/usr/bin/env python3
"""
Generate high-quality translation quality data using NLLB-200.

This script:
1. Loads facebook/nllb-200-distilled-600M (or a larger variant if available)
2. For each of our 20 languages × 19 target languages = 380 pairs:
   - Discovers false friends by translating probe words in isolation
   - Discovers idioms by translating multi-word expressions
3. Saves results to utils/quality_resources/{false_friends,idioms}.json

Usage:
    python scripts/generate_quality_data.py                          # default: distilled-600M, CPU
    python scripts/generate_quality_data.py --model facebook/nllb-200-3.3B --device cuda  # larger model
    python scripts/generate_quality_data.py --langs es fr de --device cuda  # subset only
    python scripts/generate_quality_data.py --dry-run               # show what would be generated

Output:
    Writes to utils/quality_resources/{false_friends,idioms}.json
    Existing entries are preserved; new data is merged in.
"""
import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ALL_LANGS = [
    "en", "es", "fr", "de", "pt", "it", "ja", "zh", "ru",
    "ar", "ko", "nl", "pl", "tr", "th", "vi", "hi", "sv",
    "uk", "id",
]

RESOURCES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "utils", "quality_resources"
)


def load_existing(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(data)} entries to {path}")


def dry_run_info(langs):
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "qe",
        os.path.join(os.path.dirname(__file__), "..", "utils", "quality_extractor.py"),
    )
    qe = importlib.util.module_from_spec(spec)

    class _NoOp:
        def __getattr__(self, name):
            return self
        def __call__(self, *a, **kw):
            return self
        def __iter__(self):
            return iter([])
        def __len__(self):
            return 0

    _sys.modules["torch"] = _NoOp()
    _sys.modules["transformers"] = _NoOp()
    spec.loader.exec_module(qe)

    total = len(langs) * (len(langs) - 1)
    n_words = sum(len(v) for v in qe.PROBE_WORDS.values())
    print(f"Languages: {len(langs)} → {total} direction pairs")
    print(f"Probe words per pair: {n_words}")
    print(f"Idiom probes available for: {list(qe.IDIOM_PROBES.keys())}")
    print(f"Tone probes: formal={len(qe.TONE_PROBES['formal'])}, casual={len(qe.TONE_PROBES['casual'])}")
    print()
    print("Output files:")
    print(f"  {os.path.join(RESOURCES_DIR, 'false_friends.json')}")
    print(f"  {os.path.join(RESOURCES_DIR, 'idioms.json')}")
    print()
    print(f"Each pair generates ~{n_words // 3}-{n_words} false friends and ~5-15 idioms")
    print(f"Estimated total: {total * n_words // 3}-{total * n_words} false friends, {total * 5}-{total * 15} idioms")


def main():
    parser = argparse.ArgumentParser(description="Generate quality data from NLLB-200")
    parser.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="NLLB model name (default: distilled-600M, alternatives: nllb-200-1.3B, nllb-200-3.3B)",
    )
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help=f"Language subset (default: all 20: {' '.join(ALL_LANGS)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without running",
    )
    args = parser.parse_args()

    langs = args.langs or ALL_LANGS

    if args.dry_run:
        dry_run_info(langs)
        return

    try:
        from utils.quality_extractor import generate_all_pairs_data
    except ImportError as e:
        logger.error(f"Cannot import quality extractor: {e}")
        logger.error("Ensure transformers and torch are installed:")
        logger.error("  pip install transformers torch")
        sys.exit(1)

    logger.info(f"Loading NLLB model: {args.model}")
    logger.info(f"Generating data for {len(langs)} languages ({len(langs) * (len(langs) - 1)} pairs)")

    ff_data, idiom_data = generate_all_pairs_data(
        model_name=args.model,
        device=args.device,
        langs=langs,
    )

    # Merge with any existing data
    ff_path = os.path.join(RESOURCES_DIR, "false_friends.json")
    idiom_path = os.path.join(RESOURCES_DIR, "idioms.json")

    existing_ff = load_existing(ff_path)
    existing_idiom = load_existing(idiom_path)

    existing_ff.update(ff_data)
    existing_idiom.update(idiom_data)

    save_json(ff_path, existing_ff)
    save_json(idiom_path, existing_idiom)

    logger.info("Done! Quality data generated from NLLB-200 model output.")
    logger.info(f"False friends: {len(existing_ff)} pairs")
    logger.info(f"Idioms: {len(existing_idiom)} pairs")


if __name__ == "__main__":
    main()
