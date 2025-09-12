# utils/model_versioning.py
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Make torch optional for registry operations
try:
    import torch  # type: ignore
    TORCH_VERSION = getattr(torch, "__version__", None)
except Exception:
    TORCH_VERSION = None

class ModelVersion:
    def __init__(self, model_dir: str = "models"):
        # Use absolute, repo-rooted path to avoid CWD issues
        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.model_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'models': {}, 'latest': {}}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:12]  # Use first 12 chars
    
    def register_model(self, 
                      model_path: str,
                      model_type: str,
                      metrics: Optional[Dict[str, float]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model version"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Compute version
        model_hash = self._compute_model_hash(str(model_path))
        timestamp = datetime.now().strftime("%Y%m%d")
        version = f"v1.0.{timestamp}.{model_hash}"
        
        # Create versioned filename
        versioned_name = f"{model_path.stem}_{version}{model_path.suffix}"
        versioned_path = model_path.parent / versioned_name
        
        # Copy to versioned name
        import shutil
        shutil.copy2(model_path, versioned_path)
        
        # Register in registry
        self.registry['models'][version] = {
            'version': version,
            'type': model_type,
            'path': str(versioned_path.resolve()),
            'original_path': str(model_path.resolve()),
            'hash': model_hash,
            'size_mb': model_path.stat().st_size / 1024**2,
            'created_at': datetime.now().isoformat(),
            'pytorch_version': TORCH_VERSION,
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        # Update latest mapping without overwriting other entries
        latest = self.registry.get('latest', {})
        latest[model_type] = version
        self.registry['latest'] = latest
        
        self._save_registry()
        
        return version
    
    def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a model version"""
        return self.registry['models'].get(version)
    
    def get_latest_version(self, model_type: str) -> Optional[str]:
        """Get latest version for a model type"""
        return self.registry.get('latest', {}).get(model_type)
    
    def list_versions(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by type"""
        versions: List[Dict[str, Any]] = []
        for version, info in self.registry['models'].items():
            if model_type is None or info['type'] == model_type:
                versions.append(info)
        
        # Sort by creation date
        versions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return versions