#!/usr/bin/env python3
"""
Guard script to verify required secrets for mobile SDK publish jobs exist.
- Checks environment variables are present (provided by workflow 'env' / 'secrets'):
  * MAVEN_USERNAME, MAVEN_PASSWORD for Android (maven-publish)
  * COCOAPODS_TRUNK_TOKEN for iOS (pod trunk)
Exits 1 if any are missing.
"""
from __future__ import annotations
import os
import sys

REQUIRED = {
    'android': ['MAVEN_USERNAME', 'MAVEN_PASSWORD'],
    'ios': ['COCOAPODS_TRUNK_TOKEN'],
}


def main() -> int:
    missing = []
    for k, keys in REQUIRED.items():
        for key in keys:
            if not os.environ.get(key):
                missing.append(key)
    if missing:
        print('Missing required publish secrets: ' + ', '.join(sorted(set(missing))), file=sys.stderr)
        return 1
    print('Publish secrets present')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())