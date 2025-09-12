#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

# Insecure defaults mirrored from utils/secrets_bootstrap.py
INSECURE_DEFAULTS = {
    "jwtsecret123",
    "changeme123",
    "a-very-secret-key-for-cookies",
    "a-super-secret-jwt-key",
    "internal-secret-token-for-service-auth",
    "use-openssl-rand-hex-32-to-generate-a-secure-key",
}

# Files/paths to scan (exclude examples and docs intentionally)
EXCLUDE_DIRS = {".git", ".github", "docs", "web", "react-native", "flutter"}
EXCLUDE_FILES = {".env.example"}

SECRET_KEYS = {
    "DECODER_JWT_SECRET",
    "COORDINATOR_JWT_SECRET",
    "COORDINATOR_SECRET",
    "COORDINATOR_TOKEN",
    "INTERNAL_SERVICE_TOKEN",
    "UTS_HMAC_KEY",
}

ENV_ASSIGN_RE = re.compile(r"^([A-Z0-9_]+)=(.*)$")


def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\0" in chunk
    except Exception:
        return True


def scan_tree(root: Path) -> int:
    violations = 0
    for p in root.rglob("*"):
        # Exclusions
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name in EXCLUDE_FILES:
            continue
        if not p.is_file() or is_binary(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Check for insecure defaults present verbatim
        for bad in INSECURE_DEFAULTS:
            if bad and bad in text:
                # allow if commented out lines likely examples; only flag for likely assignments
                lines = [ln for ln in text.splitlines() if bad in ln]
                for ln in lines:
                    if ln.strip().startswith("#"):
                        continue
                    if any(key in ln and "=" in ln for key in SECRET_KEYS):
                        print(f"[FAIL] Insecure default found in {p}: {ln.strip()}")
                        violations += 1
                        break
        
        # Ensure *_FILE is preferred where sensitive keys appear in shell env files
        if p.suffix in {".env", ""} and p.name.startswith('.') is False:
            for ln in text.splitlines():
                m = ENV_ASSIGN_RE.match(ln.strip())
                if not m: 
                    continue
                key, val = m.group(1), m.group(2)
                if key in SECRET_KEYS and val and not key.endswith("_FILE"):
                    print(f"[WARN] Consider using {key}_FILE instead of inline value in {p}")
        
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    violations = scan_tree(repo_root)
    if violations:
        print(f"Secret hygiene check failed with {violations} violation(s).")
        return 1
    print("Secret hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())