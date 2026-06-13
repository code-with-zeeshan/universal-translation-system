#!/usr/bin/env python3
"""
Fail if any python file contains `yaml.load(` which can be unsafe.
Exceptions: explicit safe loader forms or comments do not exempt the check.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"\byaml\.load\s*\(")

# Note: This is intentionally strict. If you need unsafe loading for a
# controlled reason, use yaml.safe_load or document/waive in CI policy.

def file_uses_unsafe_yaml(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    # Quick check: look for yaml.load(
    return bool(re.search(r"\byaml\s*\.\s*load\s*\(", text))


def main() -> int:
    src_dirs = [
        ROOT / "runtime",
        ROOT / "utils",
        ROOT / "tools",
        ROOT / "tests",
        ROOT / "scripts",
        ROOT / "universal-decoder-node",
        ROOT / "vocabulary",
        ROOT / "pipeline",
    ]

    offenders = []
    for base in src_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            # Skip this checker file itself
            if path.resolve() == Path(__file__).resolve():
                continue
            if file_uses_unsafe_yaml(path):
                offenders.append(str(path.relative_to(ROOT)))

    if offenders:
        print("Unsafe yaml.load usage found in:")
        for f in sorted(offenders):
            print(f" - {f}")
        print("\nUse yaml.safe_load instead.")
        return 1

    print("No unsafe yaml.load usage detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())