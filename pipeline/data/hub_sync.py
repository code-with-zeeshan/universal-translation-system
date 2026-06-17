"""Sync processed data and vocabulary to/from Hugging Face Hub.

Allows running the expensive data pipeline once (e.g., on Colab free tier),
uploading the result to HF Hub, then downloading on GPU instances for training.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _is_logged_in() -> bool:
    try:
        from huggingface_hub import HfFolder
        return bool(HfFolder.get_token())
    except Exception:
        return False


# ── File list for sync ───────────────────────────────────────────────

def _files_to_sync(processed_dir: Path, vocab_dir: Path, datasets_dir: Path) -> list[tuple[str, Path]]:
    """Return list of (repo_path, local_path) for all files to sync."""
    files = []

    # Entire datasets/ dir uploaded as datasets/ prefix
    if datasets_dir.exists():
        for fpath in datasets_dir.rglob("*"):
            if fpath.is_file():
                repo_subpath = f"datasets/{fpath.relative_to(datasets_dir)}"
                files.append((repo_subpath, fpath))

    if (processed_dir / "train_temp.txt").exists():
        files.append(("train_temp.txt", processed_dir / "train_temp.txt"))

    # Vocabulary packs
    if vocab_dir.exists():
        for fpath in vocab_dir.rglob("*"):
            if fpath.is_file():
                repo_subpath = f"vocab/{fpath.relative_to(vocab_dir)}"
                files.append((repo_subpath, fpath))

    # Pipeline state (enables resume on another machine)
    pipeline_state = processed_dir.parent.parent / "pipeline_state.json"
    if pipeline_state.exists():
        files.append(("pipeline_state.json", pipeline_state))

    return files


# ── Existence check ──────────────────────────────────────────────────

def data_exists_locally(processed_dir: Path, vocab_dir: Path, datasets_dir: Path) -> bool:
    """Check if training data, validation data, and vocab manifest exist."""
    missing = whats_missing(processed_dir, vocab_dir, datasets_dir)
    return not (missing.get("data") or missing.get("vocab"))


def whats_missing(processed_dir: Path, vocab_dir: Path, datasets_dir: Path) -> dict:
    """Return dict indicating which components are missing {'data': bool, 'vocab': bool}."""
    train_ok = (datasets_dir / "train_final.txt").exists()
    val_ok = (datasets_dir / "val_final.txt").exists()
    data_ok = train_ok and val_ok
    vocab_ok = (vocab_dir / "manifest.json").exists()
    if not train_ok:
        logger.info("Missing: %s", datasets_dir / "train_final.txt")
    if not val_ok:
        logger.info("Missing: %s", datasets_dir / "val_final.txt")
    if not vocab_ok:
        logger.info("Missing: %s", vocab_dir / "manifest.json")
    return {"data": not data_ok, "vocab": not vocab_ok}


# ── Upload ───────────────────────────────────────────────────────────

def upload_processed_data(
    repo_id: str,
    processed_dir: Path,
    vocab_dir: Path,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    datasets_dir: Optional[Path] = None,
) -> int:
    """Upload processed data and vocabulary packs to HF Hub.

    Returns number of files uploaded.
    """
    if not _is_logged_in() and not token:
        logger.warning("Not logged into HF Hub. Set HF_TOKEN or login first.")
        return 0

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token)

    if datasets_dir is None:
        datasets_dir = processed_dir
    files = _files_to_sync(processed_dir, vocab_dir, datasets_dir)
    if not files:
        logger.warning("No files found to upload in %s or %s", processed_dir, vocab_dir)
        return 0

    uploaded = 0
    for repo_path, local_path in files:
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
            uploaded += 1
            logger.info("  Uploaded: %s", repo_path)
        except HfHubHTTPError as e:
            logger.error("  Failed to upload %s: %s", repo_path, e)

    logger.info("Uploaded %d/%d files to %s", uploaded, len(files), repo_id)
    return uploaded


# ── Download ─────────────────────────────────────────────────────────

def download_processed_data(
    repo_id: str,
    processed_dir: Path,
    vocab_dir: Path,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    datasets_dir: Optional[Path] = None,
) -> bool:
    """Download processed data and vocabulary from HF Hub.

    Returns True if files were downloaded, False if no download needed.
    """
    if datasets_dir is None:
        datasets_dir = processed_dir
    if data_exists_locally(processed_dir, vocab_dir, datasets_dir):
        logger.info("Data and vocab already exist locally — skipping download.")
        return False

    if not _is_logged_in() and not token:
        logger.warning("Not logged into HF Hub. Set HF_TOKEN or login first.")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)

    # Download datasets/* via snapshot (includes train_final.txt, val_final.txt, etc.)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns="datasets/*",
            local_dir=str(datasets_dir.parent),
            token=token,
        )
        # If snapshot created datasets/ subdir, move contents up
        snapshot_datasets = datasets_dir.parent / "datasets"
        if snapshot_datasets.exists() and snapshot_datasets != datasets_dir:
            import shutil
            for fpath in snapshot_datasets.iterdir():
                dest = datasets_dir / fpath.name
                if fpath.is_dir():
                    shutil.copytree(str(fpath), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(fpath), str(dest))
            shutil.rmtree(str(snapshot_datasets))
    except HfHubHTTPError:
        logger.warning("No datasets/ directory found in repo %s", repo_id)

    _ensure_file(repo_id, "pipeline_state.json", processed_dir.parent.parent / "pipeline_state.json", token)

    # Download vocab/* via snapshot
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns="vocab/*",
            local_dir=str(vocab_dir.parent),
            token=token,
        )
        # Rename vocab/ subdir if snapshot created it
        snapshot_vocab = vocab_dir.parent / "vocab"
        if snapshot_vocab.exists() and snapshot_vocab != vocab_dir:
            import shutil
            if vocab_dir.exists():
                shutil.rmtree(vocab_dir)
            shutil.move(str(snapshot_vocab), str(vocab_dir))
    except HfHubHTTPError:
        logger.warning("No vocab/ directory found in repo %s", repo_id)

    success = data_exists_locally(processed_dir, vocab_dir, datasets_dir)
    if success:
        logger.info("Downloaded data and vocab from %s", repo_id)
    else:
        logger.warning("Some files still missing after download from %s", repo_id)
    return success


def _ensure_file(repo_id: str, repo_path: str, local_path: Path, token: Optional[str] = None, repo_type: str = "dataset"):
    """Download a single file if it doesn't exist locally."""
    if local_path.exists():
        return
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=repo_type,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
            token=token,
        )
        logger.info("  Downloaded: %s", repo_path)
    except HfHubHTTPError:
        logger.warning("  File not found in repo: %s", repo_path)
