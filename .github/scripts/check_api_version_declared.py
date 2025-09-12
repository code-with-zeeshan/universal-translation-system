#!/usr/bin/env python3
"""
Fail CI if coordinator/decoder FastAPI info.version differ or drift from core.apiVersion.
- Reads /openapi.json from both services when URLs provided via env (optional)
- Also checks imported constants API_VERSION matches core.apiVersion locally
Exits non-zero on mismatch.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from urllib.request import urlopen


def fetch_openapi_version(url: str) -> str:
    url = url.rstrip('/') + '/openapi.json'
    with urlopen(url, timeout=5) as resp:
        data = json.load(resp)
    return str(data.get('info', {}).get('version', ''))


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'version-config.json').read_text(encoding='utf-8'))
    core_api = str(cfg.get('core', {}).get('apiVersion', ''))
    if not core_api:
        print('core.apiVersion missing in version-config.json', file=sys.stderr)
        return 2

    # Local constant check
    sys.path.insert(0, str(root))
    try:
        from utils.constants import API_VERSION  # type: ignore
    except Exception as e:
        print(f'Failed to import API_VERSION: {e}', file=sys.stderr)
        return 3
    if str(API_VERSION) != core_api:
        print(f'Local API_VERSION ({API_VERSION}) != core.apiVersion ({core_api})', file=sys.stderr)
        return 4

    # Optional live services check
    coord = os.environ.get('COORDINATOR_URL')
    dec = os.environ.get('DECODER_URL')
    if coord and dec:
        try:
            cv = fetch_openapi_version(coord)
            dv = fetch_openapi_version(dec)
            if not cv or not dv:
                print('OpenAPI version missing from one service', file=sys.stderr)
                return 5
            if cv != dv:
                print(f'API version mismatch: coordinator={cv} decoder={dv}', file=sys.stderr)
                return 6
            if cv != core_api:
                print(f'Service API version ({cv}) != core.apiVersion ({core_api})', file=sys.stderr)
                return 7
        except Exception as e:
            print(f'Failed to fetch live OpenAPI: {e}', file=sys.stderr)
            return 8

    print('API version checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())