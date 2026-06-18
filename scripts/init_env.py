#!/usr/bin/env python3
"""
init_env.py — Initialize .env from .env.example with auto-generated secrets.

Usage:
    python scripts/init_env.py                        # default: general role
    python scripts/init_env.py --role coordinator     # pre-fill coordinator secrets
    python scripts/init_env.py --role decoder         # pre-fill decoder secrets
    python scripts/init_env.py --role general         # only UTS_HMAC_KEY
    python scripts/init_env.py --all                  # generate ALL possible secrets
    python scripts/init_env.py --check                # check existing .env for weak values

What it does:
    1. Copies .env.example -> .env
    2. Replaces placeholder secrets with cryptographically random values
    3. Optionally generates an RSA-2048 keypair for RS256 JWT auth
    4. Validates the result
"""

import argparse
import os
import re
import secrets
import shutil
import string
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"
SECRETS_DIR = ROOT / "output" / "secrets"

# All derivable secrets — (env_var, min_len, description)
# REDIS_PASSWORD excluded: handled separately with rand_password()
SECRETS = [
    ("UTS_HMAC_KEY",            32, "HMAC signing key -- ALWAYS required"),
    ("COORDINATOR_SECRET",      32, "Coordinator cookie/session key"),
    ("COORDINATOR_JWT_SECRET",  32, "JWT signing secret (HS256)"),
    ("COORDINATOR_TOKEN",       32, "Admin API token"),
    ("INTERNAL_SERVICE_TOKEN",  32, "Service-to-service auth token"),
    ("DECODER_JWT_SECRET",      32, "Decoder JWT secret"),
]

# Other password-like secrets
OTHER_SECRETS = [
    "REDIS_PASSWORD",
    "GRAFANA_ADMIN_PASSWORD",
]

# Env vars that MUST be present at runtime (flagged by --check)
REQUIRED_VARS = {
    "UTS_HMAC_KEY": "HMAC key for secure serialization",
    "COORDINATOR_JWT_SECRET": "JWT secret for coordinator (or set RS256 keys)",
    "COORDINATOR_SECRET": "Cookie/ session key for coordinator",
    "COORDINATOR_TOKEN": "Admin token for coordinator",
    "DECODER_JWT_SECRET": "JWT secret for decoder",
    "INTERNAL_SERVICE_TOKEN": "Service-to-service auth token",
    "REDIS_PASSWORD": "Redis password (if REDIS_URL used)",
    "HF_TOKEN": "HuggingFace token (if HF Hub sync used)",
}

PLACEHOLDER_PATTERNS = [
    r"use-openssl-rand-hex-\d+-to-generate-a-secure-[a-z-]+",
    r"replace-with-strong-random",
    r"paste-private-key-here",
    r"paste-public-key-here",
    r"hf_xxx",
]

INSECURE_LITERALS = {
    "jwtsecret123",
    "changeme123",
    "a-very-secret-key-for-cookies",
    "a-super-secret-jwt-key",
    "internal-secret-token-for-service-auth",
    "use-openssl-rand-hex-32-to-generate-a-secure-key",
}


def rand_str(min_len: int = 32) -> str:
    byte_len = (min_len + 1) // 2 + 8
    return secrets.token_hex(byte_len)[:min_len]


def rand_password(length: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def gen_rsa_keypair() -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend(),
    )
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub


def is_placeholder(val: str) -> bool:
    for pat in PLACEHOLDER_PATTERNS:
        if re.fullmatch(pat, val.strip()):
            return True
    return val.strip() in INSECURE_LITERALS


def replace_var(content: str, var: str, new_val: str) -> str:
    """Replace FIRST occurrence of VAR=<value> (commented or active), uncommenting it.

    Handles:
        VAR=value         ->  VAR=newval
        # VAR=value       ->  VAR=newval    (uncomments)
        VAR = value       ->  VAR = newval  (preserves spacing)
    """
    pattern = re.compile(
        r"^(#\s*)??(" + re.escape(var) + r"\s*=\s*)\S+.*$",
        re.MULTILINE,
    )

    def replacer(m: re.Match) -> str:
        assign = m.group(2)
        return assign + new_val

    return pattern.sub(replacer, content, count=1)


def role_secrets(role: str, all_flag: bool) -> set:
    if all_flag:
        return {s[0] for s in SECRETS} | set(OTHER_SECRETS)
    s = {"UTS_HMAC_KEY"}
    if role == "coordinator":
        s |= {"COORDINATOR_SECRET", "COORDINATOR_JWT_SECRET",
              "COORDINATOR_TOKEN", "INTERNAL_SERVICE_TOKEN",
              "REDIS_PASSWORD"}
    if role == "decoder":
        s |= {"DECODER_JWT_SECRET", "INTERNAL_SERVICE_TOKEN",
              "REDIS_PASSWORD"}
    if role in ("coordinator", "decoder"):
        s.add("GRAFANA_ADMIN_PASSWORD")
    return s


def write_secret_files(content: str):
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    var_map = {
        "COORDINATOR_JWT_SECRET": "coordinator_jwt_secret.txt",
        "COORDINATOR_SECRET": "coordinator_secret.txt",
        "COORDINATOR_TOKEN": "coordinator_token.txt",
        "DECODER_JWT_SECRET": "decoder_jwt_secret.txt",
        "INTERNAL_SERVICE_TOKEN": "internal_service_token.txt",
    }
    for var, filename in var_map.items():
        m = re.search(rf"^{var}=(\S+)", content, re.MULTILINE)
        if m:
            path = SECRETS_DIR / filename
            path.write_text(m.group(1) + "\n")
            path.chmod(0o600)


def run(args: argparse.Namespace) -> int:
    if args.check:
        return check_env()

    if not ENV_EXAMPLE.exists():
        print(f"Error: {ENV_EXAMPLE} not found")
        return 1

    to_fill = role_secrets(args.role, args.all)

    # Write to temp file first for atomicity
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=ENV_FILE.parent, prefix=".env.tmp.", delete=False,
    )
    try:
        content = ENV_EXAMPLE.read_text()

        # Replace each secret in SECRETS list
        for var, min_len, desc in SECRETS:
            if var in to_fill:
                new_val = rand_str(min_len)
                content = replace_var(content, var, new_val)
                print(f"  {var} generated ({len(new_val)} hex chars)")

        # Handle password-style secrets
        for var in OTHER_SECRETS:
            if var in to_fill:
                new_val = rand_password()
                content = replace_var(content, var, new_val)
                print(f"  {var} generated ({len(new_val)} chars)")

        # Generate RSA keypair if requested
        if args.rsa or args.all:
            try:
                priv, pub = gen_rsa_keypair()
                SECRETS_DIR.mkdir(parents=True, exist_ok=True)
                (SECRETS_DIR / "jwt_private_key.pem").write_text(priv)
                (SECRETS_DIR / "jwt_public_key.pem").write_text(pub)
                print(f"  RSA-2048 keypair saved to {SECRETS_DIR}/")
                print(f"    Set JWT_PRIVATE_KEY_FILE={SECRETS_DIR/'jwt_private_key.pem'}")
                print(f"    Set JWT_PUBLIC_KEY_PATH={SECRETS_DIR/'jwt_public_key.pem'}")
                priv_flat = priv.replace("\n", "\\n")
                pub_flat = pub.replace("\n", "\\n")
                content = replace_var(content, "JWT_PRIVATE_KEY", priv_flat)
                content = replace_var(content, "JWT_PUBLIC_KEY", pub_flat)
            except ImportError:
                print("  [!] cryptography not installed; skipping RSA key generation")
                print("      Install: pip install cryptography")

        tmp.write(content)
        tmp.close()
        shutil.move(tmp.name, ENV_FILE)
    except BaseException:
        if not tmp.closed:
            tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise

    print(f"\n  Saved to {ENV_FILE}")

    write_secret_files(content)
    print(f"  Secret files updated in {SECRETS_DIR}/")

    return 0


def check_env() -> int:
    if not ENV_FILE.exists():
        print(f"No .env file found at {ENV_FILE}")
        return 1

    content = ENV_FILE.read_text()
    weak = []
    for var, desc in REQUIRED_VARS.items():
        m = re.search(rf"^[^#]*{var}=(\S+)", content, re.MULTILINE)
        if m:
            val = m.group(1)
            if is_placeholder(val):
                weak.append((var, val, desc))
        else:
            weak.append((var, "(missing or commented)", desc))

    if not weak:
        print("All secrets look strong.")
        return 0

    print(f"Found {len(weak)} weak/missing secrets:\n")
    for var, val, desc in weak:
        print(f"  [!] {var} = {val[:56]}")
        print(f"       ({desc})")
    print()
    print("Fix with: python scripts/init_env.py --role <your-role>")
    print("          python scripts/init_env.py --all   (fill everything)")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Initialize .env from template with secure secrets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--role", choices=["general", "coordinator", "decoder"],
                        default="general")
    parser.add_argument("--all", action="store_true",
                        help="Generate ALL possible secrets")
    parser.add_argument("--rsa", action="store_true",
                        help="Also generate RSA-2048 keypair for RS256 JWT")
    parser.add_argument("--check", action="store_true",
                        help="Check existing .env for weak values")
    args = parser.parse_args()

    try:
        sys.exit(run(args))
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
