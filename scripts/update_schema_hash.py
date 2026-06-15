#!/usr/bin/env python3
"""
Compute combined API schema hash and update version-config.json core.schemaHash.
- Scans docs/schemas, openapi.(yml|yaml|json), and proto/*.proto
- Writes the sha256 to core.schemaHash and sets core.schemaHashVersion=core.apiVersion
"""
import hashlib
import json
from pathlib import Path
import sys

from _shared import sha256_file, find_schema_files

ROOT = Path(__file__).resolve().parents[1]
VCFG = ROOT / 'version-config.json'


def main() -> int:
    if not VCFG.exists():
        print('version-config.json not found', file=sys.stderr)
        return 2
    cfg = json.loads(VCFG.read_text(encoding='utf-8'))

    files = find_schema_files()
    if not files:
        print('No schema files found; nothing to update')
        return 0

    comb = hashlib.sha256()
    for p in files:
        try:
            h = sha256_file(p)
            comb.update(bytes.fromhex(h))
        except Exception:
            pass
    schema_hash = comb.hexdigest()

    core = cfg.get('core', {})
    api_ver = core.get('apiVersion')
    if not api_ver:
        print('core.apiVersion missing; set it before locking schema hash', file=sys.stderr)
        return 3

    prev = core.get('schemaHash') or ''
    core['schemaHash'] = schema_hash
    core['schemaHashVersion'] = api_ver
    cfg['core'] = core

    VCFG.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print(f'Updated core.schemaHash to {schema_hash} for apiVersion {api_ver}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())