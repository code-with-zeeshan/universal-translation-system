#!/usr/bin/env python3
"""
Simple runtime smoke check to ensure API versions align.
- Fetches /openapi.json from coordinator and decoder and compares 'info.version'
Usage:
  python scripts/runtime_api_version_check.py --coordinator http://localhost:9000 --decoder http://localhost:8000
Exits non-zero on mismatch.
"""
from __future__ import annotations
import argparse
import sys
import json
import urllib.request


def fetch_openapi_version(url: str) -> str:
    if url.endswith('/'):
        url = url[:-1]
    endpoint = url + '/openapi.json'
    with urllib.request.urlopen(endpoint, timeout=5) as resp:
        data = json.load(resp)
    info = data.get('info', {})
    return str(info.get('version', ''))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--coordinator', required=True)
    ap.add_argument('--decoder', required=True)
    args = ap.parse_args()

    try:
        cv = fetch_openapi_version(args.coordinator)
        dv = fetch_openapi_version(args.decoder)
    except Exception as e:
        print(f'Failed to fetch openapi: {e}', file=sys.stderr)
        return 2

    if not cv or not dv:
        print('OpenAPI version missing in one of the services', file=sys.stderr)
        return 3

    if cv != dv:
        print(f'API version mismatch: coordinator={cv} decoder={dv}', file=sys.stderr)
        return 4

    print(f'API version OK: {cv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())