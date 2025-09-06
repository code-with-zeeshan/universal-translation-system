"""
utils/jwks_utils.py

Centralized helpers to build JWKS from environment/config and compute diffs.
Used by coordinator and decoder to avoid duplication and to update metrics.
"""
from __future__ import annotations

import os
import base64
from typing import List, Dict, Tuple
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

try:
    # Optional: only available where monitoring is bundled
    from monitoring.metrics import JWKS_RELOADS_SUCCESS, JWKS_RELOADS_FAILURE, JWKS_KEYS
except Exception:  # pragma: no cover - metrics may not be available in all contexts
    JWKS_RELOADS_SUCCESS = None
    JWKS_RELOADS_FAILURE = None
    JWKS_KEYS = None


def _b64url_uint(val: int) -> str:
    raw = val.to_bytes((val.bit_length() + 7) // 8, byteorder="big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _derive_kid_from_spki(public_key: rsa.RSAPublicKey) -> str:
    spki = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    h = hashes.Hash(hashes.SHA256(), backend=default_backend())
    h.update(spki)
    return base64.urlsafe_b64encode(h.finalize()).rstrip(b"=").decode("ascii")


def _load_pems_from_env(env: os._Environ = os.environ) -> Tuple[List[str], List[str]]:
    """Load public key PEM strings and optional key IDs from env/file paths.

    Supports:
    - JWT_PUBLIC_KEY: '||'-separated PEM strings
    - JWT_PUBLIC_KEY_PATH: '||'-separated file paths to PEM files
    - JWT_KEY_IDS: '||'-separated key IDs aligned with the keys order
    """
    public_keys_raw = env.get("JWT_PUBLIC_KEY") or ""
    public_keys_from_files: List[str] = []
    key_ids_raw = env.get("JWT_KEY_IDS") or ""

    key_paths = env.get("JWT_PUBLIC_KEY_PATH") or ""
    for path in [p for p in key_paths.split("||") if p.strip()]:
        try:
            with open(path, "rb") as f:
                public_keys_from_files.append(f.read().decode("utf-8"))
        except Exception:
            # Defer logging to caller; keep utility simple
            continue

    pems = [p for p in (public_keys_raw.split("||") + public_keys_from_files) if p.strip()]
    key_ids = [k for k in key_ids_raw.split("||") if k.strip()]
    return pems, key_ids


def build_jwks_from_env(component: str, env: os._Environ = os.environ) -> List[Dict[str, str]]:
    """Build a JWKS list from env and update metrics.

    Returns list of JWK dicts with fields: kty, kid, use, alg, n, e.
    Updates JWKS metrics (success/failure and key count) when metrics are available.
    """
    pems, key_ids = _load_pems_from_env(env)
    jwks: List[Dict[str, str]] = []
    try:
        for idx, pem in enumerate(pems):
            try:
                key = serialization.load_pem_public_key(pem.encode("utf-8"), backend=default_backend())
                if not isinstance(key, rsa.RSAPublicKey):
                    continue
                numbers = key.public_numbers()
                n_b64 = _b64url_uint(numbers.n)
                e_b64 = _b64url_uint(numbers.e)
                kid = key_ids[idx] if idx < len(key_ids) else _derive_kid_from_spki(key)
                jwks.append({
                    "kty": "RSA",
                    "kid": kid,
                    "use": "sig",
                    "alg": "RS256",
                    "n": n_b64,
                    "e": e_b64,
                })
            except Exception:
                # skip invalid entries
                continue
        if JWKS_KEYS is not None:
            try:
                JWKS_KEYS.labels(component=component).set(len(jwks))
            except Exception:
                pass
        if JWKS_RELOADS_SUCCESS is not None:
            try:
                JWKS_RELOADS_SUCCESS.labels(component=component).inc()
            except Exception:
                pass
    except Exception:
        if JWKS_RELOADS_FAILURE is not None:
            try:
                JWKS_RELOADS_FAILURE.labels(component=component).inc()
            except Exception:
                pass
        # Return whatever we built so far on failure (could be empty)
        return jwks
    return jwks


def diff_kids(old_keys: List[Dict[str, str]], new_keys: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """Compute added and removed kid lists between old and new JWKS sets."""
    old_kids = {k.get("kid") for k in old_keys}
    new_kids = {k.get("kid") for k in new_keys}
    added = sorted(new_kids - old_kids)
    removed = sorted(old_kids - new_kids)
    return added, removed