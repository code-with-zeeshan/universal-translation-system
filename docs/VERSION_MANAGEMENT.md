# Version Management

Centralized semantic versioning for all components — encoder, decoder, coordinator, and all 5 SDKs.

## Architecture

```
version-config.json  ←  single source of truth
        │
        ├── scripts/version_manager.py  — CLI tool (show, update, check, release)
        ├── scripts/runtime_api_version_check.py — runtime endpoint verification
        └── .github/workflows/version-check.yml — CI gate on PRs
```

### `version-config.json`

Defines versions for 10 components with compatibility constraints:

| Component | Version File(s) |
|---|---|
| `core` | — |
| `encoder` | — |
| `decoder` | — |
| `coordinator` | — |
| `android-sdk` | `build.gradle` (versionName/versionCode) |
| `ios-sdk` | `.podspec` (s.version), `Package.swift` |
| `python-package` | `setup.py`, `pyproject.toml`, `universal-decoder-node/__init__.py` |
| `web-sdk` | `package.json` |
| `react-native-sdk` | `package.json` |
| `root-python` | `setup.py`, `__init__.py` |

Each component specifies `compatibleWith` as semver ranges:

```json
"encoder": {
  "version": "1.0.0",
  "compatibleWith": {
    "decoder": ">=1.0.0 <2.0.0",
    "coordinator": ">=1.0.0 <2.0.0"
  }
}
```

## CLI Reference

```bash
# Show all component versions
uts tools --version
# Equivalent: python scripts/version_manager.py show

# Update a component version (auto-patches all build files)
python scripts/version_manager.py update encoder 1.1.0

# Check compatibility between all components
python scripts/version_manager.py check

# Validate version-config.json against actual build files
python scripts/version_manager.py validate

# Print compatibility matrix
python scripts/version_manager.py matrix

# Create a release (updates all SDK versions + git tag)
python scripts/version_manager.py release 1.1.0
```

## Runtime API Version Check

After deploying, verify the running services agree on API version:

```bash
python scripts/runtime_api_version_check.py \
  --coordinator http://localhost:5100 \
  --decoder http://localhost:8001
```

Exits non-zero if `/openapi.json` versions mismatch.

## CI/CD Gate

`.github/workflows/version-check.yml` runs on every PR that touches version-related files. It:
1. Validates `version-config.json` structure
2. Prints the compatibility matrix
3. Checks that all SDK build files match the declared versions

## Model Version Registry

Separate from component versioning — `utils/model_versioning.py` provides:

- `ModelVersion.register()` — register a model checkpoint with SHA-256 hash + HMAC signature
- `ModelVersion.download()` — download a specific version
- `ModelVersion.promote()` — promote a version to production
- `ModelVersion.rollback()` — revert to a previous version
- Hugging Face Hub integration for distributed access

Model versions are stored in a JSON registry file with file-locking for concurrent safety.

## Best Practices

1. Follow semver: `MAJOR.MINOR.PATCH`
   - MAJOR — breaking API changes
   - MINOR — new features, backward compatible
   - PATCH — bug fixes
2. Run `validate` and `matrix` before any release
3. Keep `version-config.json` as the single source of truth — never edit build files directly
4. Update version on every PR that changes API surface or SDK interfaces
5. Use `runtime_api_version_check.py` in your deployment pipeline
