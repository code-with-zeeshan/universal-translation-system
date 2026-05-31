# CI/CD Guide for Universal Translation System

This guide describes the recommended CI/CD workflows for building, testing, and deploying the system.

---

## Overview
- **Encoder Core (C++):** Built via `scripts/build_encoder_core.sh`, Docker multi-stage, and/or Kubernetes Job.
- **Decoder (Cloud):** Built as a FastAPI/uvicorn Docker container, deployed to Kubernetes or cloud VMs.
- **SDKs:** Artifacts for Android, iOS, Flutter, React Native, and Web are built and published for app integration.
- **Version Management:** `version-config.json` + `scripts/version_manager.py` for component semver.

---

## Encoder Core CI/CD

### Docker Build (for CI or Edge Packaging)
```bash
docker build -f docker/encoder.Dockerfile -t universal-encoder-core:latest .
```

### Native Build Script
```bash
bash scripts/build_encoder_core.sh   # Supports Linux x86_64, macOS universal, Android NDK, iOS
```

---

## Decoder CI/CD

### Docker Build
```bash
docker build -f docker/decoder.Dockerfile -t universal-decoder:latest .
```

---

## SDK CI/CD

### Android/iOS/Flutter
- Build using platform-native tools (Gradle, Xcode, Flutter).
- Link against encoder core artifact.
- Publish to Maven, CocoaPods, pub.dev.
- Reference: docs/SDK_PUBLISHING.md

### React Native/Web
- Build with Node.js, TypeScript, bundlers.
- Publish to npm.
- Workflows: `.github/workflows/web-npm-publish.yml`, `.github/workflows/sdk-publish.yml`

---

## Example CI Pipeline (GitHub Actions)

```yaml
name: Universal Translation CI
on: [push, pull_request]
jobs:
  build-encoder:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Encoder Core
        run: docker build -f docker/encoder.Dockerfile -t universal-encoder-core:latest .
      - name: Extract Artifact
        run: |
          id=$(docker create universal-encoder-core:latest)
          docker cp $id:/usr/lib/libuniversal_encoder_core.so ./artifacts/
          docker rm $id
      - uses: actions/upload-artifact@v3
        with:
          name: encoder-core
          path: ./artifacts/

  build-decoder:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Decoder
        run: docker build -f docker/decoder.Dockerfile -t universal-decoder:latest .
      - uses: actions/upload-artifact@v3
        with:
          name: decoder-image
          path: .

  verify-schema-hash:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Verify schema hash
        run: |
          pip install -r requirements/base.txt
          python scripts/update_schema_hash.py
          git diff --exit-code version-config.json || {
            echo "version-config.json schemaHash drifted. Run 'python scripts/update_schema_hash.py'";
            exit 1;
          }

  build-flutter-sdk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Flutter SDK
        run: |
          cd sdk/flutter/universal_translation_sdk
          flutter pub get
          flutter build aar
      - uses: actions/upload-artifact@v3
        with:
          name: flutter-sdk
          path: sdk/flutter/universal_translation_sdk/build/
```

---

## Best Practices
- Use versioned Docker tags for production releases (`v*.*.*`).
- Verify `version-config.json` schema hash in CI.
- Store build artifacts in a central repository.
- Automate tests for all SDKs and server components.
- Use Kubernetes namespaces and resource limits.
- Monitor deployments with health checks and Prometheus metrics.

---

## See Also
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- SDK READMEs in each SDK folder
- [API.md](./API.md)
- [CI_BUILD_UPLOAD.md](./CI_BUILD_UPLOAD.md)
