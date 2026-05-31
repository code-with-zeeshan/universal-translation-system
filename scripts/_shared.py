"""Shared utility functions for scripts."""
import hashlib
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_schema_files() -> List[Path]:
    candidates: List[Path] = []
    schemas_root = ROOT / "docs" / "schemas"
    if schemas_root.exists():
        for p in schemas_root.rglob('*'):
            if p.is_file() and (
                p.suffix.lower() in {'.yaml', '.yml', '.json'} or p.suffix.lower() == '.proto'
            ):
                candidates.append(p)
    for p in [ROOT / 'openapi.yaml', ROOT / 'openapi.yml', ROOT / 'openapi.json']:
        if p.exists():
            candidates.append(p)
    proto_dir = ROOT / 'proto'
    if proto_dir.exists():
        candidates.extend(list(proto_dir.rglob('*.proto')))
    return candidates
