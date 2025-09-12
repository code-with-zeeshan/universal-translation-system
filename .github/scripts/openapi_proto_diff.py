#!/usr/bin/env python3
"""
Optional deeper schema compatibility checks.
- If previous schema artifacts are present in .artifacts/schemas_prev, compare hashes
- If live schemas exist under docs/schemas, proto/ or openapi.{json,yaml}, compare
- Fails on breaking changes when MAJOR/MINOR bump not detected in core.apiVersion
This is a lightweight diff: uses sha256 per file and aggregates with file-name ordering.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUR = []
PREV = []


def list_schema_files(base: Path):
    files = []
    # OpenAPI/JSON/YAML
    for p in [base / 'openapi.yaml', base / 'openapi.yml', base / 'openapi.json']:
        if p.exists():
            files.append(p)
    docs = base / 'docs' / 'schemas'
    if docs.exists():
        for p in docs.rglob('*'):
            if p.is_file() and (p.suffix.lower() in {'.yaml', '.yml', '.json'} or p.suffix.lower() == '.proto'):
                files.append(p)
    proto_dir = base / 'proto'
    if proto_dir.exists():
        files += list(proto_dir.rglob('*.proto'))
    return sorted(set(files))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(p.name.encode('utf-8'))
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def main() -> int:
    cfg = json.loads((ROOT / 'version-config.json').read_text(encoding='utf-8'))
    core_api = str(cfg.get('core', {}).get('apiVersion', '0.0.0'))
    prev_hash = str(cfg.get('core', {}).get('schemaHash') or '')
    prev_ver = str(cfg.get('core', {}).get('schemaHashVersion') or '0.0.0')

    current = list_schema_files(ROOT)
    if not current:
        print('No schema files found; skipping deeper diff')
        return 0
    curr_hash = combined_hash(current)
    print(f'Current schema set hash: {curr_hash}')

    if not prev_hash:
        print('core.schemaHash not set; recommend setting it to lock the current schema')
        return 0

    if curr_hash == prev_hash:
        print('Schema hash unchanged')
        return 0

    # Hash changed; require major/minor bump over schemaHashVersion
    try:
        import semver  # type: ignore
        cur = semver.VersionInfo.parse(core_api)
        prev = semver.VersionInfo.parse(prev_ver)
    except Exception as e:
        print(f'Failed to parse versions: {e}', file=sys.stderr)
        return 2

    if cur.major > prev.major or (cur.major == prev.major and cur.minor > prev.minor):
        print('Schema changed and API version bump detected (OK)')
        return 0

    print(f'Schema changed but core.apiVersion ({core_api}) did not bump minor/major over previous ({prev})', file=sys.stderr)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())