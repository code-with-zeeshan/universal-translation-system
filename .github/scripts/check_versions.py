import json, re, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
config_path = root / 'version-config.json'
config = json.loads(config_path.read_text(encoding='utf-8'))

errors = []

# Decoder node python
setup_py = root / 'universal-decoder-node' / 'setup.py'
if setup_py.exists():
    m = re.search(r'version\s*=\s*[\"\']([^\"\']+)[\"\']', setup_py.read_text(encoding='utf-8'))
    if m and m.group(1) != config['components']['python-package']['version']:
        errors.append(f"decoder setup.py: {m.group(1)} != {config['components']['python-package']['version']}")
pyproject = root / 'universal-decoder-node' / 'pyproject.toml'
if pyproject.exists():
    m = re.search(r'^version\s*=\s*[\"\']([^\"\']+)[\"\']', pyproject.read_text(encoding='utf-8'), re.M)
    if m and m.group(1) != config['components']['python-package']['version']:
        errors.append(f"decoder pyproject: {m.group(1)} != {config['components']['python-package']['version']}")

# Web SDK
web_pkg = root / 'web' / 'universal-translation-sdk' / 'package.json'
if web_pkg.exists():
    v = json.loads(web_pkg.read_text(encoding='utf-8')).get('version')
    c = config['components'].get('web-sdk',{}).get('version')
    if c and v != c:
        errors.append(f"web package.json: {v} != {c}")

# RN SDK
rn_pkg = root / 'react-native' / 'UniversalTranslationSDK' / 'package.json'
if rn_pkg.exists():
    v = json.loads(rn_pkg.read_text(encoding='utf-8')).get('version')
    c = config['components'].get('react-native-sdk',{}).get('version')
    if c and v != c:
        errors.append(f"react-native package.json: {v} != {c}")

# Root setup.py
root_setup = root / 'setup.py'
if root_setup.exists():
    m = re.search(r'version\s*=\s*[\"\']([^\"\']+)[\"\']', root_setup.read_text(encoding='utf-8'))
    c = config['components'].get('root-python',{}).get('version')
    if m and c and m.group(1) != c:
        errors.append(f"root setup.py: {m.group(1)} != {c}")

if errors:
    print('Version mismatches:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('Versions consistent with version-config.json')