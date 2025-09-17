from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List


def load_version_info(path: str = 'version-config.json') -> Dict[str, Any]:
    """Safely load version info from version-config.json.
    Returns a dict with possible keys: version, apiVersion, schemaHash.
    """
    try:
        core = json.loads(Path(path).read_text(encoding='utf-8')).get('core', {})
        return {
            'version': core.get('version'),
            'apiVersion': core.get('apiVersion'),
            'schemaHash': core.get('schemaHash'),
        }
    except Exception:
        return {'version': None, 'apiVersion': None, 'schemaHash': None}


def jwks_readiness(env: os._Environ = os.environ, jwks_keys: List[Dict[str, Any]] | None = None) -> bool:
    """Check JWKS readiness only when RS256 is configured via env.
    If RS256 not configured, returns True.
    """
    jwks_keys = jwks_keys or []
    rs256_configured = bool(env.get('JWT_PUBLIC_KEY') or env.get('JWT_PUBLIC_KEY_PATH'))
    return (len(jwks_keys) > 0) if rs256_configured else True


def build_ready_payload(component: str, version: Dict[str, Any], checks: Dict[str, Any]) -> Dict[str, Any]:
    """Create a consistent readiness payload.
    - component: 'decoder' | 'coordinator' | etc.
    - version: dict from load_version_info()
    - checks: dict of boolean or structured checks
    """
    def _all_true(val):
        if isinstance(val, dict):
            # Only consider booleans inside dict; ignore counters/metadata
            bool_values = [v for v in val.values() if isinstance(v, bool)]
            return all(bool_values) if bool_values else True
        return bool(val)

    ready = all(_all_true(v) for v in checks.values())
    return {
        'ready': bool(ready),
        'component': component,
        'version': version.get('version'),
        'apiVersion': version.get('apiVersion'),
        'schemaHash': version.get('schemaHash'),
        'checks': checks,
    }