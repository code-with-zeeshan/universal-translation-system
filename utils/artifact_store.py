# utils/artifact_store.py
"""
Artifact store utilities for pulling models, vocabulary packs, and adapters
from Hugging Face Hub on demand and ensuring they exist locally.

Environment variables:
- HF_HUB_REPO_ID: e.g., your-username/universal-translation-system
- HF_TOKEN (optional): token for private repos or uploads
- HF_HUB_REVISION (optional): branch/tag/commit to target (default: main)
- MODELS_DIR (optional): default local models dir (default: models)
- VOCABS_DIR (optional): default local vocab dir (default: vocabs)
- ADAPTERS_DIR (optional): default local adapters dir (default: models/adapters)

Typical usage:

from utils.artifact_store import ArtifactStore
store = ArtifactStore()
# Ensure encoder model exists locally (by filename inside repo)
store.ensure_model("production/encoder.onnx")
# Ensure vocabulary pack exists
store.ensure_vocab_pack("latin")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except Exception as e:  # pragma: no cover
    HfApi = None
    hf_hub_download = None
    list_repo_files = None

logger = logging.getLogger(__name__)


@dataclass
class StoreConfig:
    repo_id: str
    token: Optional[str]
    revision: str
    models_dir: Path
    vocabs_dir: Path
    adapters_dir: Path


class ArtifactStore:
    def __init__(self, config: Optional[StoreConfig] = None):
        if config is None:
            repo_id = os.environ.get("HF_HUB_REPO_ID")
            if not repo_id:
                raise ValueError("HF_HUB_REPO_ID must be set to use ArtifactStore")
            token = os.environ.get("HF_TOKEN")
            revision = os.environ.get("HF_HUB_REVISION", "main")
            models_dir = Path(os.environ.get("MODELS_DIR", "models"))
            vocabs_dir = Path(os.environ.get("VOCABS_DIR", "vocabs"))
            adapters_dir = Path(os.environ.get("ADAPTERS_DIR", str(models_dir / "adapters")))
            config = StoreConfig(
                repo_id=repo_id,
                token=token,
                revision=revision,
                models_dir=models_dir,
                vocabs_dir=vocabs_dir,
                adapters_dir=adapters_dir,
            )
        self.cfg = config
        for p in [self.cfg.models_dir, self.cfg.vocabs_dir, self.cfg.adapters_dir]:
            p.mkdir(parents=True, exist_ok=True)

        if HfApi is None:
            logger.warning("huggingface_hub is not installed. Artifact downloads will be unavailable.")

    def _download_file(self, repo_path: str, local_path: Path) -> Path:
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is not available. Install it to download artifacts.")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from HF: {self.cfg.repo_id}:{repo_path} -> {local_path}")
        downloaded_path = hf_hub_download(
            repo_id=self.cfg.cfg.repo_id if hasattr(self.cfg, 'cfg') else self.cfg.repo_id,  # defensive
            filename=repo_path,
            token=self.cfg.token,
            revision=self.cfg.revision,
            repo_type="model",
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        # Ensure final path
        dl = Path(downloaded_path)
        if dl != local_path:
            # If HF placed file with same name, move/rename into desired local path
            try:
                if dl.exists():
                    dl.replace(local_path)
            except Exception:
                pass
        return local_path

    def ensure_model(self, relative_repo_path: str) -> Path:
        """Ensure a model file (under models/) exists locally, downloading if needed.
        Example relative_repo_path: "models/production/encoder.onnx" or "production/encoder.onnx".
        """
        # Normalize path: accept with or without leading "models/"
        repo_path = relative_repo_path.replace("\\", "/")
        if not repo_path.startswith("models/"):
            repo_path = f"models/{repo_path}"
        local_path = self.cfg.models_dir / Path(repo_path).relative_to("models")
        if local_path.exists():
            return local_path
        return self._download_file(repo_path, local_path)

    def ensure_adapter(self, adapter_name: str, filename: Optional[str] = None) -> Path:
        """Ensure a language adapter exists locally. If filename is not provided,
        we try "models/adapters/{adapter_name}.bin" in the repo.
        """
        if filename is None:
            repo_path = f"models/adapters/{adapter_name}.bin"
        else:
            repo_path = filename if filename.startswith("models/") else f"models/{filename}"
        local_path = self.cfg.adapters_dir / Path(repo_path).relative_to("models/adapters")
        if local_path.exists():
            return local_path
        return self._download_file(repo_path, local_path)

    def ensure_vocab_pack(self, pack_name: str, version: Optional[str] = None) -> Path:
        """Ensure a vocabulary pack file exists locally.
        Defaults to vocabs/{pack_name}_v{version}.msgpack if version is provided,
        otherwise tries any {pack_name}_v*.msgpack by listing repo files.
        """
        if version:
            filename = f"{pack_name}_v{version}.msgpack"
            repo_path = f"vocabs/{filename}"
            local_path = self.cfg.vocabs_dir / filename
            if local_path.exists():
                return local_path
            return self._download_file(repo_path, local_path)

        # No version specified: try to find the newest by listing repo files
        if list_repo_files is None:
            # Fallback to a default name
            candidate = self.cfg.vocabs_dir / f"{pack_name}.msgpack"
            if candidate.exists():
                return candidate
            raise RuntimeError("list_repo_files unavailable. Specify version or install huggingface_hub.")
        files = list_repo_files(self.cfg.repo_id, repo_type="model", revision=self.cfg.revision, token=self.cfg.token)
        # Filter matching pack files
        candidates: List[str] = [f for f in files if f.startswith(f"vocabs/{pack_name}_v") and f.endswith(".msgpack")]
        if not candidates:
            raise FileNotFoundError(f"No vocabulary pack found in repo for {pack_name}")
        # Simple sort: lexicographic by filename
        candidates.sort(reverse=True)
        repo_path = candidates[0]
        local_path = self.cfg.vocabs_dir / Path(repo_path).name
        if local_path.exists():
            return local_path
        return self._download_file(repo_path, local_path)

    def ensure_for_language_pair(self, source_lang: str, target_lang: str, adapter: Optional[str] = None) -> None:
        """Ensure necessary artifacts for a language pair are available locally."""
        # Ensure vocab by target language pack family
        pack_hint = {
            'zh': 'cjk', 'ja': 'cjk', 'ko': 'cjk',
            'ar': 'arabic', 'hi': 'devanagari', 'ru': 'cyrillic', 'uk': 'cyrillic', 'th': 'thai'
        }.get(target_lang, 'latin')
        try:
            self.ensure_vocab_pack(pack_hint)
        except Exception as e:
            logger.warning(f"Failed to ensure vocab pack {pack_hint}: {e}")
        # Optional adapter
        if adapter:
            try:
                self.ensure_adapter(adapter)
            except Exception as e:
                logger.warning(f"Failed to ensure adapter {adapter}: {e}")