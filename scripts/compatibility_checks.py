#!/usr/bin/env python3
"""
Additional API/schema/tokenizer compatibility checks beyond SemVer.
- Validates that required components exist in version-config.json
- Checks bidirectional compatibility using VersionManager (supplements validate)
- Performs lightweight vocabulary/tokenizer checks when metadata exists
- Validates API schemas (OpenAPI/Swagger, Protobuf) when present
- Optionally validates ONNX model opset/runtime when models are present
- Emits non-zero exit on hard failures; warns on missing optional metadata
"""
from __future__ import annotations
import json
import sys
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
VCFG = ROOT / "version-config.json"


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_schema_files() -> List[Path]:
    candidates: List[Path] = []
    schemas_root = ROOT / "docs" / "schemas"
    if schemas_root.exists():
        for p in schemas_root.rglob('*'):
            if p.is_file() and (
                p.suffix.lower() in {'.yaml', '.yml', '.json'} or p.suffix.lower() == '.proto'
            ):
                candidates.append(p)
    # also consider top-level api specs if any
    for p in [ROOT / 'openapi.yaml', ROOT / 'openapi.yml', ROOT / 'openapi.json']:
        if p.exists():
            candidates.append(p)
    proto_dir = ROOT / 'proto'
    if proto_dir.exists():
        candidates.extend(list(proto_dir.rglob('*.proto')))
    return candidates


def validate_openapi_like(path: Path) -> List[str]:
    warnings: List[str] = []
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return [f"Failed to read {path}: {e}"]
    # quick checks for presence of openapi/swagger and info.version
    has_openapi = re.search(r'\bopenapi\b|\bswagger\b', text, re.IGNORECASE) is not None
    info_version = re.search(r'info\s*:\s*(?:\n|.)*?version\s*[:=]\s*(["\']?)([^"\'\n]+)\1', text, re.IGNORECASE)
    if not has_openapi:
        warnings.append(f"{path}: missing 'openapi' or 'swagger' key")
    if not info_version:
        warnings.append(f"{path}: missing info.version")
    return warnings


def validate_protobuf(path: Path) -> List[str]:
    warnings: List[str] = []
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return [f"Failed to read {path}: {e}"]
    if 'syntax = "proto3";' not in text and "syntax='proto3';" not in text:
        warnings.append(f"{path}: missing proto3 syntax declaration")
    if re.search(r'\bpackage\s+\w+\s*;', text) is None:
        warnings.append(f"{path}: missing package declaration")
    return warnings


def validate_onnx_models() -> List[str]:
    warnings: List[str] = []
    onnx_paths: List[Path] = []
    for base in [ROOT / 'models', ROOT / 'cloud_decoder', ROOT / 'encoder']:
        if base.exists():
            onnx_paths.extend(list(base.rglob('*.onnx')))
    if not onnx_paths:
        return warnings  # nothing to check
    try:
        import onnx  # type: ignore
    except Exception:
        warnings.append("! ONNX models found but 'onnx' package not available; skipping deep validation")
        return warnings
    for p in onnx_paths:
        try:
            model = onnx.load(str(p))
            # opset imports can have multiple domains; default domain is ''
            opsets = {imp.domain: imp.version for imp in model.opset_import}
            default_opset = opsets.get('', max(opsets.values()) if opsets else 0)
            if default_opset is None:
                default_opset = max(opsets.values()) if opsets else 0
            if default_opset < 13:
                warnings.append(f"{p}: opset {default_opset} < 13 (min supported)")
            # Optional: IR version sanity check
            if getattr(model, 'ir_version', 0) and model.ir_version < 7:
                warnings.append(f"{p}: IR version {model.ir_version} < 7 (min recommended)")
        except Exception as e:
            warnings.append(f"Failed to parse ONNX {p}: {e}")
    return warnings


def main() -> int:
    if not VCFG.exists():
        print("✗ version-config.json not found", file=sys.stderr)
        return 2
    cfg = load_json(VCFG)
    # Enforce shared API version and vocab format policy
    core_api = cfg.get('core', {}).get('apiVersion')
    vocab_supported = str(cfg.get('vocabularies', {}).get('supported_format', '')).split('.')[0]

    # 1) Ensure critical components exist
    required = ["encoder", "decoder", "coordinator", "web-sdk", "react-native-sdk", "python-package", "root-python"]
    missing = [c for c in required if c not in cfg.get("components", {})]
    if missing:
        print(f"✗ Missing components in version-config.json: {', '.join(missing)}", file=sys.stderr)
        return 3

    # 2) Bidirectional compatibility check using VersionManager
    try:
        # Ensure repo root (parent of 'scripts') is importable
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.version_manager import VersionManager  # type: ignore
    except Exception as e:
        print(f"✗ Failed to import VersionManager: {e}", file=sys.stderr)
        return 4

    vm = VersionManager(ROOT)
    for a in required:
        v_a = vm.get_current_version(a)
        for b in required:
            if a == b:
                continue
            v_b = vm.get_current_version(b)
            fwd = vm.check_compatibility(a, v_a, b, v_b)
            rev = vm.check_compatibility(b, v_b, a, v_a)
            if not (fwd and rev):
                print(f"✗ Compatibility failure: {a} {v_a} <-> {b} {v_b}", file=sys.stderr)
                return 5

    # 3) Tokenizer/Vocabulary format checks (lightweight)
    # Optional manifest at vocabulary/manifest.json with a format_version field
    vocab_manifest = ROOT / "vocabulary" / "manifest.json"
    if vocab_manifest.exists():
        try:
            m = load_json(vocab_manifest)
            fmt = m.get("format_version")
            if not fmt:
                print("✗ vocabulary/manifest.json missing format_version", file=sys.stderr)
                return 6
            # enforce allowed major per version-config.json
            if str(fmt).split('.')[0] != vocab_supported:
                print(f"✗ Unsupported vocabulary format_version: {fmt} (supported major: {vocab_supported})", file=sys.stderr)
                return 7
        except Exception as e:
            print(f"✗ Failed reading vocabulary manifest: {e}", file=sys.stderr)
            return 8
    else:
        print("! Warning: vocabulary/manifest.json not found; skipping vocab format check")

    # 4) API schema checks (OpenAPI/Swagger/JSON + Protobuf)
    schema_files = find_schema_files()
    if schema_files:
        combined_hash = hashlib.sha256()
        hard_fail = False
        for p in schema_files:
            try:
                h = sha256_file(p)
                combined_hash.update(bytes.fromhex(h))
            except Exception:
                pass
            if p.suffix.lower() in {'.yaml', '.yml', '.json'}:
                issues = validate_openapi_like(p)
            elif p.suffix.lower() == '.proto':
                issues = validate_protobuf(p)
            else:
                issues = []
            for msg in issues:
                print(f"✗ {msg}", file=sys.stderr)
            hard_fail = hard_fail or any(msg.startswith('Failed') or 'missing' in msg for msg in issues)
        schema_hash = combined_hash.hexdigest()
        print(f"Schema set hash (sha256): {schema_hash}")
        # Enforce: if schema hash changed, require core.apiVersion bump (major/minor)
        # Compare against version-config.json core.schemaHash when present
        prev_hash = str(cfg.get('core', {}).get('schemaHash') or '')
        if prev_hash and prev_hash != schema_hash:
            # Require at least a minor bump on apiVersion (strict gate)
            try:
                import semver  # type: ignore
                current = semver.VersionInfo.parse(str(core_api or '0.0.0'))
                prev_ver = semver.VersionInfo.parse(str(cfg.get('core', {}).get('schemaHashVersion') or '0.0.0'))
                if not (current.major > prev_ver.major or (current.major == prev_ver.major and current.minor > prev_ver.minor)):
                    print(f"✗ API schema changed but core.apiVersion ({core_api}) did not bump minor/major over previous ({prev_ver})", file=sys.stderr)
                    return 9
            except Exception:
                print(f"✗ API schema changed (hash diff). Bump core.apiVersion and update core.schemaHash in version-config.json (current apiVersion: {core_api})", file=sys.stderr)
                return 9
        if not prev_hash:
            print("ℹ️ core.schemaHash not set; set it to lock current schema in version-config.json")
        if hard_fail:
            return 9
    else:
        print("! Warning: No API schema files found; skipping schema checks")

    # 5) ONNX model checks (optional; warns if onnx not installed)
    onnx_issues = validate_onnx_models()
    for msg in onnx_issues:
        if msg.startswith('! '):
            print(msg)
        else:
            print(f"✗ {msg}", file=sys.stderr)
    if any(not m.startswith('! ') for m in onnx_issues):
        return 10

    # 6) Ensure core.apiVersion is present for gating
    if not core_api:
        print("✗ core.apiVersion missing in version-config.json", file=sys.stderr)
        return 11

    print("✓ Compatibility checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())