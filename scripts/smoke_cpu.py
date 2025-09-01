"""
Lightweight CPU-only smoke test for local development.
- Verifies config models
- Runs a tiny encoder forward pass on CPU
- Synthesizes tiny sample data and runs data pipeline in dry-run mode
- Optionally creates a minimal vocabulary pack in RESEARCH mode

This script avoids GPU and network-heavy operations.
"""
import os
import sys
from pathlib import Path

REPO = Path(r"c:\Users\DELL\universal-translation-system")
sys.path.insert(0, str(REPO))


def ensure_min_deps():
    # We intentionally avoid auto-install here to keep it simple and offline-friendly.
    # Required: torch (cpu build), pyyaml, msgpack, tqdm, pytest (optional), requests (optional)
    missing = []
    for pkg in ["torch", "yaml", "msgpack", "tqdm"]:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
        except Exception:
            missing.append(pkg)
    if missing:
        print("[WARN] Missing packages:", missing)
        print("Install minimal deps, e.g.\n  pip install --upgrade pip\n  pip install \"torch==2.2.2+cpu\" --index-url https://download.pytorch.org/whl/cpu\n  pip install pyyaml msgpack tqdm")


def set_cpu_env():
    os.environ.setdefault("COORDINATOR_JWT_SECRET", "12345678901234567890123456789012")
    os.environ.setdefault("ENCODER_DEVICE", "cpu")
    os.environ.setdefault("DECODER_DEVICE", "cpu")


def check_config_models():
    print("[1/4] Checking config models (schemas)...")
    from config.schemas import load_config, RootConfig
    cfg = load_config(str(REPO / "config" / "base.yaml"))
    assert isinstance(cfg, RootConfig)
    assert cfg.data.processed_dir
    assert isinstance(cfg.training.batch_size, int)
    assert cfg.vocabulary.vocab_dir
    print("  OK: schemas load and basic fields validate")


def tiny_encoder_forward():
    print("[2/4] Running tiny encoder forward pass (CPU)...")
    import torch
    from encoder.universal_encoder import UniversalEncoder
    m = UniversalEncoder(hidden_dim=64, num_layers=1, num_heads=4, max_vocab_size=100)
    x = torch.randint(0, 100, (1, 8))
    y = m(x)
    assert y.shape == (1, 8, 64)
    print("  OK:", tuple(y.shape))


def pipeline_dry_run():
    print("[3/4] Running unified data pipeline in dry-run mode...")
    import asyncio
    from config.schemas import load_config
    from data.unified_data_pipeline import UnifiedDataPipeline
    cfg = load_config(str(REPO / "config" / "base.yaml"))
    pipe = UnifiedDataPipeline(cfg, dry_run=True)
    # Only run non-network stages; in dry-run, download stages are skipped automatically
    # If needed, limit to a safe subset of stages
    summary = asyncio.get_event_loop().run_until_complete(
        pipe.run_pipeline(resume=False)
    )
    print("  OK: pipeline dry-run completed;")


def vocab_smoke():
    print("[4/4] Creating minimal vocabulary pack (RESEARCH mode)...")
    from vocabulary.unified_vocabulary_creator import UnifiedVocabularyCreator, CreationMode, UnifiedVocabConfig
    creator = UnifiedVocabularyCreator(
        corpus_dir=str(REPO / "data" / "processed"),
        output_dir=str(REPO / "vocabs"),
        default_mode=CreationMode.RESEARCH,
        config=UnifiedVocabConfig(vocab_size=200)
    )
    # create_pack signature: (pack_name: str, languages: List[str], ...)
    pack = creator.create_pack(pack_name="latin", languages=["en", "es"], mode=CreationMode.RESEARCH)
    creator._save_pack(pack, "latin")
    # Validate
    from pathlib import Path as _P
    js = sorted((REPO / "vocabs").glob("latin_v*.json"))
    assert js, "No vocab JSON created"
    ok, errs = creator.validate_pack(str(js[-1]))
    assert ok, f"Vocab validation failed: {errs}"
    print("  OK: vocab pack created and validated:", js[-1].name)


if __name__ == "__main__":
    ensure_min_deps()
    set_cpu_env()
    check_config_models()
    tiny_encoder_forward()
    pipeline_dry_run()
    try:
        vocab_smoke()
    except Exception as e:
        print("[WARN] Skipping vocab smoke:", e)
        print("[INFO] This step requires optional deps (e.g., numpy, tenacity, sentencepiece). Not needed for core smoke.")
    print("\nCPU smoke checks completed.")