#!/usr/bin/env python3
"""
uts publish — Single source-of-truth publish to Hugging Face Hub.

Usage:
  ./uts tools --upload your-org/repo --checkpoint checkpoints/*/best_model.pt

What it does:
  1. Finds the best checkpoint
  2. Splits encoder & decoder state dicts into separate files
  3. Copies to models/production/
  4. Exports encoder to ONNX (if torch.onnx available)
  5. Runs quantization pipeline
  6. Uploads everything (models, vocabs, adapters) to HF Hub
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("publish")

ROOT = Path(__file__).resolve().parent.parent


def find_best_checkpoint(checkpoint_arg: str | None) -> Path:
    """Find the best_model.pt to publish."""
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")

    # Auto-discover: find the most recent best_model.pt
    from utils.common_utils import RuntimeDirectoryManager
    ckpt_dir = RuntimeDirectoryManager().checkpoints_dir
    candidates = sorted(ckpt_dir.rglob("best_model.pt"))
    if candidates:
        chosen = candidates[-1]  # most recent
        logger.info(f"Auto-discovered checkpoint: {chosen}")
        return chosen
    raise FileNotFoundError(
        "No best_model.pt found under checkpoints/. "
        "Train a model first, or pass --checkpoint explicitly."
    )


def load_checkpoint(path: Path) -> dict:
    """Load a PyTorch checkpoint and return the state dict."""
    import torch
    logger.info(f"Loading checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def split_and_save(ckpt: dict, output_dir: Path):
    """Split encoder+decoder from checkpoint and save as separate files."""
    encoder_path = output_dir / "encoder.pt"
    decoder_path = output_dir / "decoder.pt"

    # Determine which keys belong to which component
    encoder_keys = {}
    decoder_keys = {}
    for key, val in ckpt.items():
        if key.startswith("encoder") or key.startswith("enc_"):
            encoder_keys[key] = val
        elif key.startswith("decoder") or key.startswith("dec_"):
            decoder_keys[key] = val
        elif key == "encoder_state_dict":
            encoder_keys = val
        elif key == "decoder_state_dict":
            decoder_keys = val

    # If the checkpoint has nested state dicts, use them directly
    if "encoder_state_dict" in ckpt and "decoder_state_dict" in ckpt:
        encoder_sd = ckpt["encoder_state_dict"]
        decoder_sd = ckpt["decoder_state_dict"]
    elif encoder_keys and decoder_keys:
        encoder_sd = encoder_keys
        decoder_sd = decoder_keys
    else:
        # Try loading via the evaluate_model infrastructure
        logger.warning("Checkpoint format not recognized — saving full checkpoint as-is")
        shutil.copy(path, encoder_path)
        shutil.copy(path, decoder_path)
        return encoder_path, decoder_path

    import torch
    torch.save(encoder_sd, encoder_path)
    torch.save(decoder_sd, decoder_path)
    logger.info(f"Saved encoder ({len(encoder_sd)} keys): {encoder_path}")
    logger.info(f"Saved decoder ({len(decoder_sd)} keys): {decoder_path}")
    return encoder_path, decoder_path


def export_onnx(encoder_path: Path, output_dir: Path, hidden_dim: int = 512):
    """Export encoder to ONNX format."""
    onnx_path = output_dir / "encoder.onnx"
    if onnx_path.exists():
        logger.info(f"ONNX already exists: {onnx_path}")
        return onnx_path

    try:
        import torch
        # Load the encoder architecture
        from runtime.encoder.universal_encoder import UniversalEncoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = UniversalEncoder(
            max_vocab_size=32000,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
        ).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.eval()

        # Export
        dummy_input = torch.randint(0, 100, (1, 64), device=device)
        torch.onnx.export(
            encoder,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["hidden_states"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "hidden_states": {0: "batch", 1: "seq"}},
            opset_version=17,
        )
        logger.info(f"Exported ONNX: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export failed (non-fatal): {e}")
        return None
    return onnx_path


def run_quantization(output_dir: Path):
    """Run the quantization pipeline on the saved encoder."""
    encoder_path = output_dir / "encoder.pt"
    master_path = output_dir / "encoder_master.pt"
    if encoder_path.exists() and not master_path.exists():
        import shutil
        shutil.copy2(encoder_path, master_path)
        logger.info(f"Copied {encoder_path} → {master_path} for quantization")
    try:
        subprocess.run(
            [sys.executable, "-m", "pipeline.training.quantization.pipeline"],
            cwd=str(ROOT), check=True, capture_output=True, text=True,
        )
        logger.info("Quantization pipeline completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Quantization pipeline failed (non-fatal): {e.stderr[:200]}")


def upload_to_hub(repo_id: str):
    """Upload all artifacts to Hugging Face Hub."""
    upload_script = ROOT / "scripts" / "upload_artifacts.py"
    if not upload_script.exists():
        logger.error(f"Upload script not found: {upload_script}")
        return False

    logger.info(f"Uploading to HF Hub: {repo_id}")
    result = subprocess.run(
        [sys.executable, str(upload_script), "--repo-id", repo_id],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Upload complete!")
        return True
    else:
        logger.error(f"Upload failed: {result.stderr[:500]}")
        return False


def verify_artifacts(output_dir: Path) -> list[Path]:
    """List all production artifacts."""
    artifacts = sorted(output_dir.iterdir()) if output_dir.exists() else []
    logger.info(f"Production artifacts in {output_dir}/:")
    for a in artifacts:
        logger.info(f"  {a.name} ({a.stat().st_size / 1e6:.1f} MB)")
    return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Publish trained model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish most recent checkpoint
  python scripts/publish.py --repo-id your-org/uts

  # Publish specific checkpoint
  python scripts/publish.py --repo-id your-org/uts --checkpoint checkpoints/exp1/best_model.pt

  # Skip ONNX export (already done or not needed)
  python scripts/publish.py --repo-id your-org/uts --no-onnx

  # Upload only (skip split/convert)
  python scripts/publish.py --repo-id your-org/uts --upload-only
""",
    )
    parser.add_argument("--repo-id", required=True, help="HF Hub repo ID (e.g., your-org/universal-translation-system)")
    parser.add_argument("--checkpoint", help="Path to best_model.pt (auto-discovers if omitted)")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--upload-only", action="store_true", help="Skip all processing, just upload existing artifacts")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Encoder hidden dimension")
    args = parser.parse_args()

    from utils.common_utils import RuntimeDirectoryManager
    output_dir = RuntimeDirectoryManager().production_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.upload_only:
        logger.info("Upload-only mode — skipping processing")
        ok = upload_to_hub(args.repo_id)
        sys.exit(0 if ok else 1)

    # Step 1: Find checkpoint
    global ckpt_path
    ckpt_path = find_best_checkpoint(args.checkpoint)

    # Step 2: Load and split checkpoint
    ckpt = load_checkpoint(ckpt_path)
    encoder_path, decoder_path = split_and_save(ckpt, output_dir)

    # Step 3: Export to ONNX
    if not args.no_onnx:
        export_onnx(encoder_path, output_dir, args.hidden_dim)

    # Step 4: Quantize
    if not args.no_quantize:
        run_quantization(output_dir)

    # Step 5: Verify
    verify_artifacts(output_dir)

    # Step 6: Upload
    ok = upload_to_hub(args.repo_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
