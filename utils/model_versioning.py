import json
import hashlib
import hmac
import os
import fcntl
import logging
import semver
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from huggingface_hub import HfApi, hf_hub_download
    _hf_available = True
except Exception:
    _hf_available = False

try:
    import torch
    TORCH_VERSION = getattr(torch, "__version__", None)
except Exception:
    TORCH_VERSION = None

logger = logging.getLogger(__name__)


def _compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _compute_file_signature(path: Path, hmac_key: str) -> str:
    h = hmac.new(hmac_key.encode("utf-8"), digestmod=hashlib.sha256)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class ModelVersion:
    def __init__(self, model_dir: str = "models", repo_id: Optional[str] = None,
                 hmac_key: Optional[str] = None):
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.model_dir / "model_registry.json"
        self._lock_file = self.model_dir / ".model_registry.lock"
        self.repo_id = repo_id
        self.hmac_key = hmac_key or os.environ.get("UTS_HMAC_KEY", "")
        self.hf_api = HfApi() if _hf_available and repo_id else None
        self.registry = self._load_registry()

    def _acquire_lock(self, blocking: bool = True) -> Any:
        """File lock for concurrent-safe registry access."""
        fd = os.open(str(self._lock_file), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            os.close(fd)
            raise
        return fd

    def _release_lock(self, fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {"models": {}, "latest": {}, "pinned": {}, "schema_version": "2"}

    def _save_registry(self) -> None:
        fd = self._acquire_lock()
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.registry, f, indent=2)
        finally:
            self._release_lock(fd)

    def register_model(self, model_path: str, model_type: str,
                       metrics: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       semver_label: Optional[str] = None,
                       stage: str = "production") -> str:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_hash = _compute_file_hash(path)
        timestamp = datetime.now(timezone.utc).isoformat()
        version = f"{model_type}_{model_hash[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        signature = _compute_file_signature(path, self.hmac_key) if self.hmac_key else ""

        if semver_label:
            try:
                semver.VersionInfo.parse(semver_label)
            except ValueError:
                raise ValueError(f"Invalid semver: {semver_label}")

        entry = {
            "version": version,
            "semver": semver_label or "",
            "path": str(path.resolve()),
            "model_type": model_type,
            "hash": model_hash,
            "signature": signature,
            "timestamp": timestamp,
            "stage": stage,
            "framework": {"pytorch": TORCH_VERSION} if TORCH_VERSION else {},
            "metrics": metrics or {},
            "metadata": metadata or {},
        }

        fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            if model_type not in registry["models"]:
                registry["models"][model_type] = []
            registry["models"][model_type].append(entry)
            registry["latest"][model_type] = version
            if semver_label:
                registry.setdefault("semver_latest", {})[model_type] = semver_label
            self.registry = registry
            self._save_registry()
        finally:
            self._release_lock(fd)

        if self.hf_api and self.repo_id:
            try:
                self.hf_api.upload_file(path_or_fileobj=str(path), path_in_repo=f"models/{version}/model.bin",
                                        repo_id=self.repo_id)
                meta = self.hf_api.upload_file(
                    path_or_fileobj=json.dumps(entry, indent=2).encode("utf-8"),
                    path_in_repo=f"models/{version}/meta.json",
                    repo_id=self.repo_id,
                )
            except Exception as e:
                logger.warning("HF Hub upload failed: %s", e)
        return version

    def download_model(self, version: str, model_type: str,
                       target_dir: Optional[str] = None) -> Path:
        if not self.hf_api or not self.repo_id:
            raise RuntimeError("HF Hub not configured; cannot download")
        target = Path(target_dir or self.model_dir / "downloads")
        target.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"models/{version}/model.bin",
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        return Path(local_path)

    def pin_version(self, model_type: str, version: Optional[str] = None,
                    semver_label: Optional[str] = None) -> str:
        fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            if "pinned" not in registry:
                registry["pinned"] = {}
            if version:
                registry["pinned"][model_type] = version
                self.registry = registry
                self._save_registry()
                return version
            if semver_label:
                for entry in registry["models"].get(model_type, []):
                    if entry.get("semver") == semver_label:
                        registry["pinned"][model_type] = entry["version"]
                        self.registry = registry
                        self._save_registry()
                        return entry["version"]
                raise ValueError(f"No model with semver {semver_label} for {model_type}")
            raise ValueError("Provide version or semver_label")
        finally:
            self._release_lock(fd)

    def get_model(self, model_type: str, version: Optional[str] = None,
                  use_pinned: bool = False) -> Optional[Dict[str, Any]]:
        if use_pinned:
            pinned = self.registry.get("pinned", {}).get(model_type)
            if pinned:
                version = pinned
        if model_type not in self.registry["models"]:
            return None
        models = self.registry["models"][model_type]
        if version:
            for m in models:
                if m["version"] == version:
                    return m
            return None
        latest = self.registry["latest"].get(model_type)
        if latest:
            for m in models:
                if m["version"] == latest:
                    return m
        return models[-1] if models else None

    def list_versions(self, model_type: str,
                      stage: Optional[str] = None) -> List[Dict[str, Any]]:
        versions = self.registry["models"].get(model_type, [])
        if stage:
            versions = [v for v in versions if v.get("stage") == stage]
        return versions

    def promote_model(self, model_type: str, version: str,
                      to_stage: str = "production") -> None:
        fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            for entry in registry["models"].get(model_type, []):
                if entry["version"] == version:
                    entry["stage"] = to_stage
                    entry["promoted_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self.registry = registry
            self._save_registry()
        finally:
            self._release_lock(fd)

    def rollback(self, model_type: str, version: str) -> str:
        entry = self.get_model(model_type, version)
        if not entry:
            raise ValueError(f"Version {version} not found for {model_type}")
        fd = self._acquire_lock()
        try:
            registry = self._load_registry()
            registry["latest"][model_type] = version
            registry.setdefault("rollback_history", {}).setdefault(model_type, []).append({
                "from": registry.get("pinned", {}).get(model_type, registry["latest"].get(model_type)),
                "to": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self.registry = registry
            self._save_registry()
        finally:
            self._release_lock(fd)
        logger.info("Rolled back %s to %s", model_type, version)
        return version

    def get_registry(self) -> Dict[str, Any]:
        return dict(self.registry)
