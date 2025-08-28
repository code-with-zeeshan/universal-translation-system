# System Architecture

## Overview

The Universal Translation System uses an edge–cloud split to keep client apps small while providing high‑quality, scalable translations. Text is encoded on device into compact embeddings, then decoded in the cloud by a pool of GPU‑powered decoder nodes, orchestrated by an advanced coordinator with health checks, load balancing, metrics, and security.

## Architecture Diagram

```mermaid
flowchart LR
    %% Clients / SDKs
    subgraph SDKs
        A1[Android]
        A2[iOS]
        A3[Flutter]
        A4[React Native]
        A5[Web (WASM)]
    end

    %% Edge: Universal Encoder + Vocab Packs
    subgraph Edge (Client)
        E1[Universal Encoder]
        E2[Vocabulary Packs\n(2–4MB per language)]
        E1 --- E2
    end

    %% Cloud: Coordinator and Decoder Pool
    subgraph Cloud
        C1[Advanced Coordinator\n- Least-loaded routing\n- Health checks\n- Auth (JWT)\n- Metrics]
        C2[(Redis)\n(optional)]
        C3[Decoder Pool\n(Litserve + PyTorch)]
        M1[Monitoring: Prometheus/Grafana]
        T1[Tracing: OpenTelemetry]

        C1 --- C2
        C1 -->|/health, /metrics| M1
        C3 -->|/metrics| M1
        C1 --> T1
        C3 --> T1
    end

    %% Flow
    A1 & A2 & A3 & A4 & A5 -->|Text| E1
    E1 -->|LZ4 + MsgPack Embeddings| C1
    C1 -->|Route to healthy node| C3
    C3 -->|Translation| C1
    C1 -->|Result| A1 & A2 & A3 & A4 & A5
```

## Components

### 1) Universal Encoder (Edge/Client)
- **Platforms:** Android, iOS, Flutter, React Native, Web
- **Implementation:**
  - Android/iOS/Flutter: native C++ core via FFI (`encoder_core`)
  - React Native: native modules with fallback to cloud
  - Web: TypeScript SDK with optional WebAssembly edge encoding and cloud fallback
- **Vocabulary:** Dynamic packs (2–4MB/language), groups (latin, cjk, etc.)
- **Output:** Compressed embeddings (LZ4 + MsgPack) typically a few KB per request

### 2) Vocabulary Packs
- **Latin:** ~3MB (covers many European languages)
- **CJK:** ~4MB (Chinese, Japanese, Korean)
- **Other groups:** 1–4MB each; dynamically downloaded on demand

### 3) Decoder Nodes (Cloud)
- **Implementation:** PyTorch models served via Litserve
- **Architecture:** 6‑layer transformer with cross‑attention and adapter support
- **Packaging:** `universal-decoder-node` (standalone node, CLI & library)
- **Runtime:** GPU-accelerated (e.g., T4, V100, 3090, A100)

### 4) Advanced Coordinator
- **Role:** Gatekeeper between encoders and decoder pool
- **Key features:**
  - Least‑loaded routing across healthy decoders
  - Health checks via `/health` and background probes
  - Prometheus metrics exposure for pool and coordinator
  - Authentication (JWT) for admin actions and dashboard
  - Circuit breaker support and robust error handling
  - Redis integration available for distributed coordination
  - Web dashboard with real‑time charts (Chart.js)
- **Flow:** Encoders call `/decode`; coordinator selects a healthy node and proxies; activity visible via dashboard and metrics

### 5) Monitoring & Observability
- **Metrics:** `prometheus-client` integrated; scrape config included
- **Dashboards:** Grafana dashboards in `monitoring/grafana/dashboards`
- **System metrics:** CPU/GPU/Memory via `GPUtil`, `psutil`
- **Tracing (optional):** OpenTelemetry packages included

## Data Flow

1. App loads required vocabulary pack(s) if not cached
2. Encoder converts text → embeddings on device (or API fallback)
3. Embeddings are compressed with LZ4 and serialized with MsgPack
4. Coordinator `/decode` receives request and validates/authenticates if needed
5. Coordinator routes to the least‑loaded healthy decoder
6. Decoder generates translation and returns to coordinator
7. Coordinator returns translation to the client

## SDK Matrix (Edge/Cloud)

| SDK            | Edge Encoding | Cloud Decoding | Native/FFI | Notes |
|----------------|---------------|----------------|------------|-------|
| Android        | Yes           | Yes            | Yes        | JNI bindings to C++ encoder core |
| iOS            | Yes           | Yes            | Yes        | Swift + C++ interoperability |
| Flutter        | Yes           | Yes            | Yes        | FFI to native encoder core |
| React Native   | Optional      | Yes            | Yes        | Native modules with cloud fallback |
| Web            | Optional      | Yes            | WASM       | WASM edge encoding with cloud fallback |

## Configuration & Tooling

- Config helpers live in `scripts/`:
  1. `scripts/config_wizard.py` – interactive creation, hardware‑aware suggestions
  2. `scripts/validate_config.py` – schema, references, consistency, GPU checks
  3. Auto‑detection logic used by training utilities for best defaults
- Example usage:
```bash
python scripts/config_wizard.py
python scripts/validate_config.py config/my_config.yaml --check-references --check-consistency --suggest-improvements
```

## Security
- JWT support for protected endpoints and admin actions
- Optional HTTPS middleware and security headers
- Rate limiting and circuit breaker patterns available

## Dependencies (Key)
- Core: `torch`, `transformers`, `sentencepiece`, `tokenizers`
- Serialization/Compression: `msgpack`, `lz4`, `zstandard`
- Serving: `litserve`, `fastapi`, `uvicorn`
- Monitoring: `prometheus-client`, `GPUtil`, `psutil`, OpenTelemetry
- Integration: `requests`, `httpx`, `redis`

## Repository Mapping
- `encoder/` – Python encoder logic, adapters, training helpers
- `encoder_core/` – C++ encoder core + headers and examples
- `cloud_decoder/` – Litserve decoder service (Docker/K8s ready)
- `universal-decoder-node/` – Standalone decoder node package (CLI + utils)
- `coordinator/` – Advanced coordinator, load balancing, circuit breaker
- `monitoring/` – Prometheus, Grafana dashboards, and metrics utilities
- `vocabulary/` – Vocabulary creation and management utilities
- `web/universal-translation-sdk/` – Web SDK with WASM build
- `react-native/UniversalTranslationSDK/` – RN SDK with native bridges
- `flutter/universal_translation_sdk/` – Flutter SDK with FFI

## Ports & Endpoints

- **Decoder Node**
  - Port: `8000` (Kubernetes defaults)
  - Endpoints: `/decode` (translation), `/health` (liveness/readiness), `/metrics` (Prometheus)
- **Coordinator**
  - Port: `5100` (Kubernetes defaults)
  - Endpoints: `/decode` (entrypoint), `/health`, `/metrics`, admin endpoints as configured
- **Notes**
  - Probes and services are configured under `kubernetes/` manifests.
  - If HTTPS is enforced at the app layer, ensure load balancers forward scheme headers to avoid redirect loops.

## Notes
- All services expose `/metrics` for Prometheus scraping
- Embedding payloads are compact and designed for low‑bandwidth scenarios
- WASM builds enable true edge encoding in modern browsers with fallback to cloud when unavailable