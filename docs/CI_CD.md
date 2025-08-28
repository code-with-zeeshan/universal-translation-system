# CI/CD Guide for Universal Translation System

This guide describes the recommended CI/CD workflows for building, testing, and deploying the Universal Translation System, including encoder/decoder containers, SDK artifacts, and cloud infrastructure.

---

## Overview
- **Encoder Core (C++):** Built via Docker multi-stage and/or Kubernetes Job, outputs shared library for SDKs (Android, iOS, Flutter, etc.).
- **Decoder (Cloud):** Built as a Litserve-based Docker container, deployed to Kubernetes or cloud VMs.
- **SDKs:** Artifacts for Android, iOS, Flutter, React Native, and Web are built and published for app integration.

---

## Encoder Core CI/CD

### Docker Build (for CI or Edge Packaging)
```bash
docker build -f docker/encoder.Dockerfile -t universal-encoder-core:latest .
# The resulting image contains /usr/lib/libuniversal_encoder_core.so and headers
```

### Kubernetes Job (for Artifact Storage)
- Use `kubernetes/encoder-build.yaml` and `encoder-artifacts-pvc.yaml` to build and store the encoder core in a shared volume.
- Artifacts can be picked up by SDK build jobs or published to an artifact repository.

---

## Decoder CI/CD

### Docker Build
```bash
# Current Dockerfile path
docker build -f docker/decoder.Dockerfile -t universal-decoder:latest .
```

### Kubernetes Deployment
- Use `kubernetes/decoder-deployment.yaml` and `decoder-service.yaml` for cloud deployment.
- Supports GPU scheduling, scaling, and health checks.

---

## SDK CI/CD

### Android/iOS/Flutter
- Build SDKs using platform-native tools (Gradle, Xcode, Flutter).
- Link against the encoder core artifact (from Docker image or K8s PVC).
- Publish to internal or public package repositories (Maven, CocoaPods, pub.dev).
- Publishing reference: docs/SDK_PUBLISHING.md

### React Native/Web
- Build and test using Node.js, TypeScript, and bundlers (Webpack, Vite, etc.).
- Publish to npm or internal registry.
- Web SDK GitHub Actions: .github/workflows/web-npm-publish.yml
- Android/iOS SDK GitHub Actions: .github/workflows/sdk-publish.yml

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
        run: docker build -f cloud_decoder/Dockerfile -t universal-decoder:latest .
      - uses: actions/upload-artifact@v3
        with:
          name: decoder-image
          path: .

  build-flutter-sdk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Flutter SDK
        run: |
          cd flutter/universal_translation_sdk
          flutter pub get
          flutter build aar # or build ios-framework
      - uses: actions/upload-artifact@v3
        with:
          name: flutter-sdk
          path: flutter/universal_translation_sdk/build/

  # Add similar jobs for Android, iOS, React Native, Web
```

---

## Best Practices
- Use versioned Docker tags for production releases.
- Store build artifacts in a central repository or artifact store.
- Automate tests for all SDKs and server components.
- Use Kubernetes namespaces and resource limits for isolation.
- Monitor deployments with health checks and Prometheus metrics.

---

## See Also
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- SDK READMEs in each SDK folder
- [API.md](./API.md) for endpoints used by SDKs and coordinator