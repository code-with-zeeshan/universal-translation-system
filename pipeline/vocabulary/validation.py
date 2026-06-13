import json
import hashlib
import hmac
import logging
import os
import semver
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("vocabulary")

PACK_SIGNING_KEY_ENV = "UTS_VOCAB_SIGNING_KEY"


def _pack_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _pack_signature(pack: Dict[str, Any], key: Optional[str] = None) -> str:
    key = key or os.environ.get(PACK_SIGNING_KEY_ENV, "")
    if not key:
        return ""
    payload = json.dumps(pack, sort_keys=True, ensure_ascii=False)
    return hmac.new(key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def validate_pack(self, pack_path: str) -> Tuple[bool, List[str]]:
    errors = []
    pack_file = Path(pack_path)

    if not pack_file.exists():
        return False, ["Pack file does not exist"]

    try:
        if pack_file.suffix == '.json':
            with open(pack_file, 'r') as f:
                pack = json.load(f)
        elif pack_file.suffix == '.msgpack':
            with open(pack_file, 'rb') as f:
                pack = msgpack.unpackb(f.read(), strict_map_key=False)
        else:
            return False, ["Unsupported format"]

        required = ['name', 'version', 'languages', 'tokens', 'subwords', 'special_tokens', 'metadata']
        for field in required:
            if field not in pack:
                errors.append(f"Missing field: {field}")

        if 'version' in pack:
            try:
                semver.VersionInfo.parse(pack['version'])
            except ValueError:
                errors.append(f"Invalid semver version: {pack['version']}")

        if 'special_tokens' in pack:
            for token in ['<pad>', '<unk>', '<s>', '</s>']:
                if token not in pack['special_tokens']:
                    errors.append(f"Missing special token: {token}")

        if 'metadata' in pack and 'total_tokens' in pack['metadata']:
            reported = pack['metadata']['total_tokens']
            actual = sum(
                len(pack.get(key, {}))
                for key in ['tokens', 'subwords', 'special_tokens']
            )
            if reported != actual:
                errors.append(f"Token count mismatch: {reported} vs {actual}")

        if 'signature' in pack and pack['signature']:
            key = os.environ.get(PACK_SIGNING_KEY_ENV, "")
            if key:
                expected = _pack_signature({k: v for k, v in pack.items() if k != 'signature'}, key)
                if pack['signature'] != expected:
                    errors.append(f"Signature mismatch (pack may be tampered)")

        if 'metadata' in pack:
            meta = pack['metadata']
            if 'compatible_decoder' in meta:
                req = meta['compatible_decoder']
                logger.info(f"Pack {pack.get('name')} requires decoder {req}")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"Error loading pack: {e}"]


def compare_packs(self, pack1_path: str, pack2_path: str) -> Dict[str, Any]:
    def _load_any(fp):
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"Pack file not found: {fp}")
        if p.suffix == '.msgpack':
            with open(p, 'rb') as f:
                return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        with open(p, 'r') as f:
            return json.load(f)

    pack1 = _load_any(pack1_path)
    pack2 = _load_any(pack2_path)

    v1 = semver.VersionInfo.parse(pack1.get('version', '0.0.0'))
    v2 = semver.VersionInfo.parse(pack2.get('version', '0.0.0'))

    comparison = {
        'pack1_name': pack1['name'],
        'pack2_name': pack2['name'],
        'pack1_version': str(v1),
        'pack2_version': str(v2),
        'version_diff': "major" if v1.major != v2.major else ("minor" if v1.minor != v2.minor else "patch"),
        'compatible': v1.major == v2.major,
        'token_overlap': len(set(pack1['tokens'].keys()) & set(pack2['tokens'].keys())),
        'subword_overlap': len(set(pack1['subwords'].keys()) & set(pack2['subwords'].keys())),
        'pack1_unique_tokens': len(set(pack1['tokens'].keys()) - set(pack2['tokens'].keys())),
        'pack2_unique_tokens': len(set(pack2['tokens'].keys()) - set(pack1['tokens'].keys())),
        'size_difference_mb': abs(
            pack1['metadata']['size_mb'] - pack2['metadata']['size_mb']
        ),
    }
    return comparison


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(IOError)
)
def _save_pack(self, pack: Dict[str, Any], pack_name: str):
    version = pack['version']
    signing_key = os.environ.get(PACK_SIGNING_KEY_ENV, "")
    if signing_key:
        pack['signature'] = _pack_signature(pack, signing_key)
    json_path = self.output_dir / f'{pack_name}_v{version}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    msgpack_path = self.output_dir / f'{pack_name}_v{version}.msgpack'
    with open(msgpack_path, 'wb') as f:
        f.write(msgpack.packb(pack))
    logger.info(f"Saved pack v{version} to {json_path} and {msgpack_path}")
    _update_manifest(self.output_dir, pack_name, version)


def _update_manifest(output_dir: Path, pack_name: str, version: str) -> None:
    manifest_path = output_dir / 'manifest.json'
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    if 'packs' not in manifest:
        manifest['packs'] = {}
    if pack_name not in manifest['packs']:
        manifest['packs'][pack_name] = {'versions': [], 'latest': ''}
    if version not in manifest['packs'][pack_name]['versions']:
        manifest['packs'][pack_name]['versions'].append(version)
    manifest['packs'][pack_name]['latest'] = version
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def _cleanup_temp_files(self, *files):
    for file_path in files:
        try:
            if Path(file_path).exists():
                os.remove(file_path)
                logger.debug(f"Cleaned up {file_path}")
        except OSError as e:
            logger.warning(f"Could not clean up {file_path}: {e}")


def _get_pack_version(self, pack_name: str, bump: str = "minor") -> str:
    existing_versions = []
    for file in self.output_dir.glob(f'{pack_name}_v*.json'):
        version_part = file.stem.split('_v')[-1]
        try:
            semver.VersionInfo.parse(version_part)
            existing_versions.append(version_part)
        except ValueError:
            continue

    if not existing_versions:
        return "1.0.0"

    latest = sorted(existing_versions, key=lambda v: semver.VersionInfo.parse(v))[-1]
    v = semver.VersionInfo.parse(latest)

    if bump == "major":
        return f"{v.major + 1}.0.0"
    elif bump == "minor":
        return f"{v.major}.{v.minor + 1}.0"
    else:
        return f"{v.major}.{v.minor}.{v.patch + 1}"
