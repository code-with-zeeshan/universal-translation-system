import os
import time
import argparse
import secrets
import sys
from pathlib import Path
from typing import Optional

from utils.credential_manager import credential_manager

DEFAULT_EXPIRY_DAYS = 90  # Recommended rotation interval


def rotate_hs256(key_name: str, expiry_days: int = DEFAULT_EXPIRY_DAYS) -> str:
    new_secret = secrets.token_urlsafe(48)
    credential_manager.set(key_name, new_secret, store_in="config")
    expiry_ts = int(time.time()) + expiry_days * 86400
    env_var = _cred_key_to_env(key_name)
    print(f"export {env_var}_EXPIRY={expiry_ts}")
    return new_secret


def generate_rs256_pair() -> tuple[str, str]:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
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


def _cred_key_to_env(key: str) -> str:
    mapping = {
        "coordinator_jwt_secret": "COORDINATOR_JWT_SECRET",
        "decoder_jwt_secret": "DECODER_JWT_SECRET",
        "coordinator_secret": "COORDINATOR_SECRET",
        "coordinator_token": "COORDINATOR_TOKEN",
        "internal_service_token": "INTERNAL_SERVICE_TOKEN",
        "uts_hmac_key": "UTS_HMAC_KEY",
    }
    return mapping.get(key, key.upper())


def rotate_all() -> None:
    keys = [
        "coordinator_jwt_secret",
        "decoder_jwt_secret",
        "coordinator_secret",
        "coordinator_token",
        "internal_service_token",
        "uts_hmac_key",
    ]
    print("Rotating all secrets...")
    for key in keys:
        try:
            rotate_hs256(key)
            print(f"  rotated {key}")
        except Exception as e:
            print(f"  FAILED {key}: {e}", file=sys.stderr)
    print("Done. Set the printed _EXPIRY env vars in your deployment.")


def main():
    ap = argparse.ArgumentParser(description="Rotate secrets with expiry tracking")
    ap.add_argument("--type", choices=["hs256", "rs256", "all"], default="hs256")
    ap.add_argument("--key", help="credential key name for HS256 (e.g., coordinator_jwt_secret)")
    ap.add_argument("--kid", help="Key ID for RS256 rotation")
    ap.add_argument("--set-env", action="store_true", help="Set env variables in current process for preview")
    ap.add_argument("--expiry-days", type=int, default=DEFAULT_EXPIRY_DAYS, help="Expiry in days")
    args = ap.parse_args()

    if args.type == "all":
        rotate_all()
        return

    if args.type == "hs256":
        if not args.key:
            ap.error("--key is required for hs256 rotation")
        new_secret = rotate_hs256(args.key, expiry_days=args.expiry_days)
        env_var = _cred_key_to_env(args.key)
        print(f"HS256 secret rotated. Key={args.key}, env={env_var}")
        print(f"Set {env_var}_EXPIRY (printed above) to enable expiry monitoring.")
        if args.set_env:
            os.environ["UTS_" + args.key.upper()] = new_secret
    else:
        priv_pem, pub_pem = generate_rs256_pair()
        kid = args.kid or f"key-{secrets.token_hex(4)}"
        print("Generated RS256 keypair. Next steps:")
        print(f"  1) kid={kid}")
        print("  2) Add private key to JWT_PRIVATE_KEY or mount via *_FILE.")
        print("  3) Append public key to JWT_PUBLIC_KEY or add to JWKS.")
        print("  4) Add kid to JWT_KEY_IDS.")
        print("  5) Keep old keys for grace period, then remove.")
        if args.set_env:
            os.environ["JWT_PRIVATE_KEY"] = priv_pem
            os.environ["JWT_PUBLIC_KEY"] = (os.environ.get("JWT_PUBLIC_KEY", "") + ("||" if os.environ.get("JWT_PUBLIC_KEY") else "") + pub_pem)
            os.environ["JWT_KEY_IDS"] = (os.environ.get("JWT_KEY_IDS", "") + ("||" if os.environ.get("JWT_KEY_IDS") else "") + kid)
        print("Private/Public PEMs generated (not printed). Save to your secret store.")


if __name__ == "__main__":
    main()
