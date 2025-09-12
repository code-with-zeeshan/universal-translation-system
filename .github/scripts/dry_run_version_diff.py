#!/usr/bin/env python3
"""
Dry-run logic to simulate tag/version diff without publishing.
- Reads version-config.json and prints what would happen if versions changed
- Useful for PRs to preview compatibility effects
"""
from __future__ import annotations
import json, sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'version-config.json').read_text(encoding='utf-8'))
    core = cfg.get('core', {})
    comps = cfg.get('components', {})
    print('=== Dry Run: Version Matrix ===')
    print(f"core.version={core.get('version')} apiVersion={core.get('apiVersion')} schemaHash={core.get('schemaHash') or '(unset)'}")
    for name, meta in comps.items():
        v = meta.get('version')
        compat = meta.get('compatibleWith', {})
        print(f"- {name}: {v} | requires: {compat}")
    print('\nNo actions performed. Use scripts/version_manager.py validate to enforce.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())