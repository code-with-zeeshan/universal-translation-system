# tools/cloud_preflight.py
"""
Preflight checks for cloud runs. Validates presence of processed data, vocabulary packs,
model artifacts, and basic config sanity. Prints a concise checklist.

Usage:
    python -m tools.cloud_preflight
"""
from pathlib import Path
import sys
import json

from config.schemas import load_config

OK = "✅"
WARN = "⚠️"
ERR = "❌"


def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def check_processed_data(cfg) -> dict:
    base = Path(cfg.data.processed_dir)
    files = {
        "train": base / "train_final.txt",
        "val": base / "val_final.txt",
        "test": base / "test_final.txt",
    }
    present = {k: exists(v) for k, v in files.items()}
    return {"path": str(base), "present": present}


def check_vocab(cfg) -> dict:
    vocab_dir = Path(cfg.vocabulary.vocab_dir)
    packs = list(vocab_dir.glob("*_v*.msgpack")) if exists(vocab_dir) else []
    manifest = vocab_dir / "manifest.json"
    return {
        "path": str(vocab_dir),
        "pack_count": len(packs),
        "has_manifest": exists(manifest),
    }


def check_models() -> dict:
    models_dir = Path("models")
    production_best = models_dir / "production" / "best_model.pt"
    registry = models_dir / "model_registry.json"
    versioned = list((models_dir / "production").glob("*_v1.*.pt")) if exists(models_dir / "production") else []
    return {
        "path": str(models_dir),
        "has_production_best": exists(production_best),
        "has_registry": exists(registry),
        "versioned_count": len(versioned),
    }


def print_summary(cfg, data_info, vocab_info, model_info):
    print("=== Cloud Preflight Checklist ===")
    print(f"Config loaded from: {cfg.dict().get('source', 'config/base.yaml') if hasattr(cfg, 'dict') else 'config/base.yaml'}")

    # Data
    data_ok = all(data_info["present"].values())
    print(f"\nData (processed_dir={data_info['path']}): {' '+OK if data_ok else ' '+WARN}")
    for name, present in data_info["present"].items():
        print(f"  - {name}: {'present' if present else 'missing'}")

    # Vocab
    vocab_ok = vocab_info["pack_count"] > 0
    print(f"\nVocabulary (vocab_dir={vocab_info['path']}): {' '+OK if vocab_ok else ' '+WARN}")
    print(f"  - packs: {vocab_info['pack_count']}")
    print(f"  - manifest.json: {'present' if vocab_info['has_manifest'] else 'missing'}")

    # Models
    model_ok = model_info["has_production_best"] and model_info["has_registry"]
    print(f"\nModels (models_dir={model_info['path']}): {' '+OK if model_ok else ' '+WARN}")
    print(f"  - production/best_model.pt: {'present' if model_info['has_production_best'] else 'missing'}")
    print(f"  - model_registry.json: {'present' if model_info['has_registry'] else 'missing'}")
    print(f"  - versioned copies: {model_info['versioned_count']}")

    # Mount guidance
    print("\nMounts required for decoder:")
    print("  - ./models -> /app/models")
    print("  - ./vocabs -> /app/vocabs")

    # Exit code guidance
    exit_code = 0
    if not data_ok:
        exit_code = 1
    if not vocab_ok:
        exit_code = 1
    if not model_ok:
        exit_code = 1

    print("\nResult:")
    if exit_code == 0:
        print(f"All good {OK} — you can deploy and run!")
    else:
        print(f"Some items missing {WARN} — fix above before running in cloud.")
    sys.exit(exit_code)


def main():
    cfg = load_config()
    data_info = check_processed_data(cfg)
    vocab_info = check_vocab(cfg)
    model_info = check_models()
    print_summary(cfg, data_info, vocab_info, model_info)


if __name__ == "__main__":
    main()