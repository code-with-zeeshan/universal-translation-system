# tools/rotate_secrets.py
"""
CLI tool to rotate shared secrets and align JWKS.
- Supports HS256 secret rotation via CredentialManager
- Supports RS256 rotation by adding a new keypair and updating env variables
- Prints out new values; do NOT log actual secrets in production streams

Usage examples:
  python tools/rotate_secrets.py --type hs256 --key coordinator_jwt_secret
  python tools/rotate_secrets.py --type hs256 --key decoder_jwt_secret
  python tools/rotate_secrets.py --type rs256 --kid new-key-2025-01 --set-env
"""
from __future__ import annotations
import os
import argparse
import secrets
import sys
from pathlib import Path
from typing import Optional

from utils.credential_manager import credential_manager


def rotate_hs256(key_name: str) -> str:
    new_secret = secrets.token_urlsafe(48)
    credential_manager.set(key_name, new_secret, store_in="config")
    return new_secret


def generate_rs256_pair() -> tuple[str, str]:
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
    except Exception as e:
        print(f"cryptography not installed or unavailable: {e}")
        sys.exit(1)

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return priv_pem, pub_pem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["hs256", "rs256"], required=True)
    ap.add_argument("--key", help="credential key name for HS256 (e.g., coordinator_jwt_secret)")
    ap.add_argument("--kid", help="Key ID for RS256 rotation")
    ap.add_argument("--set-env", action="store_true", help="Set env variables in current process for preview")
    args = ap.parse_args()

    if args.type == "hs256":
        if not args.key:
            ap.error("--key is required for hs256 rotation")
        new_val = rotate_hs256(args.key)
        print("HS256 secret rotated. Store securely and roll out:")
        print(f"  key={args.key}\n  value=<hidden>")
        if args.set_env:
            # Set in env for immediate process usage (preview/testing only)
            os.environ["UTS_" + args.key.upper()] = new_val
    else:
        # RS256 rotation: generate new pair and suggest env updates
        priv_pem, pub_pem = generate_rs256_pair()
        kid = args.kid or f"key-{secrets.token_hex(4)}"
        print("Generated RS256 keypair. Next steps:")
        print("1) Append public key to JWT_PUBLIC_KEY or add a new path and update JWT_KEY_IDS to include:")
        print(f"   kid={kid}")
        print("2) Add private key to JWT_PRIVATE_KEY (coordinator) or mount via *_FILE.")
        print("3) Keep old keys for grace period, then remove.")
        if args.set_env:
            os.environ["JWT_PRIVATE_KEY"] = priv_pem
            # For preview only; in prod set via file/path
            os.environ["JWT_PUBLIC_KEY"] = (os.environ.get("JWT_PUBLIC_KEY", "") + ("||" if os.environ.get("JWT_PUBLIC_KEY") else "") + pub_pem)
            os.environ["JWT_KEY_IDS"] = (os.environ.get("JWT_KEY_IDS", "") + ("||" if os.environ.get("JWT_KEY_IDS") else "") + kid)
        # For safety, do not print full PEMs to stdout in real deployments.
        print("Private/Public PEMs generated (not printed). Save to your secret store.")


if __name__ == "__main__":
    main()