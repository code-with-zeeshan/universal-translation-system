# System Architecture

## Table of Contents
- [Overview](#overview)
- [End‑to‑End Architecture Diagram](#end-to-end-architecture-diagram)
- [Components (In Depth)](#components-in-depth)
- [Data Flow (Detailed)](#data-flow-detailed)
- [Deployment & Scaling](#deployment--scaling)
- [Security](#security)
- [Configuration & Versioning](#configuration--versioning)
- [Ports & Endpoints](#ports--endpoints)
- [Failure Handling & Reliability](#failure-handling--reliability)
- [Repository Mapping](#repository-mapping)
- [Sequence Diagrams](#sequence-diagrams)
- [Component-Level Diagram (Modules & Interfaces)](#component-level-diagram-modules--interfaces)
- [Deployment Diagram (Kubernetes Topology)](#deployment-diagram-kubernetes-topology)
- [Related Documentation](#related-documentation)
- [Notes](#notes)

## Overview

The Universal Translation System is an edge–cloud platform designed to keep client apps small while delivering high‑quality, scalable translations. Text is encoded on device into compact embeddings and decoded in the cloud by GPU‑powered decoder nodes. A coordinator handles routing, health checks, load balancing, security, and observability. Supporting systems include vocabulary management, artifact/model storage, CI/CD, monitoring, and Kubernetes orchestration.

---

## End‑to‑End Architecture Diagram

```mermaid
flowchart LR
    %% ===========================
    %% Clients / SDKs (Edge)
    %% ===========================
    subgraph SDKs
        A1[Android]
        A2[iOS]
        A3[Flutter]
        A4[React Native]
        A5[Web - WASM]
    end

    subgraph Edge
        E1[Universal Encoder Core\nC++/FFI + WASM]
        E2[Vocabulary Manager\nPacks: Latin/CJK/... 2-4MB]
        E3[Local Cache\nLRU + Versioned]
        E4[Compression/Serialization\nLZ4 + MsgPack]
        E1 --- E2
        E2 --- E3
        E1 --> E4
    end

    %% ===========================
    %% Network & Security
    %% ===========================
    subgraph Network
        TLS[TLS/HTTPS]
        Auth[JWT / API Keys]
        RL[Rate Limiter]
        CB[Circuit Breaker]
        TLS --- Auth
        Auth --- RL
        RL --- CB
    end

    %% ===========================
    %% Cloud Control Plane
    %% ===========================
    subgraph Coordinator
        C_API[HTTP API - decode health metrics]
        C_RT[Router - Least Loaded and Health]
        C_HC[Health Prober\nActive and Passive]
        C_MX[Metrics Exporter\nPrometheus]
        C_TR[Tracing\nOpenTelemetry]
        C_ADM[Admin and Dashboard\nChartJS]
        C_REDIS[Redis\noptional]
        C_CFG[Configuration\nconfigs and env]
        C_SEC[AuthN/Z\nJWT]

        C_API --> C_RT
        C_RT --> C_HC
        C_API --> C_SEC
        C_API --> C_MX
        C_API --> C_TR
        C_RT --> C_REDIS
        C_ADM --> C_MX
        C_CFG -. reads .-> C_API
    end

    %% ===========================
    %% Decoder Data Plane
    %% ===========================
    subgraph DecoderPool
        DN1[Decoder Node #1\nLitserve + PyTorch]
        DN2[Decoder Node #2]
        DNn[Decoder Node #N]

            MLD[Model Loader\nAdapters]
            VLM[Runtime Vocab Access]
            MXS[Metrics Exporter]
            HCS[Health Endpoint]

        DN1 --- MLD
        DN1 --- VLM
        DN1 --- MXS
        DN1 --- HCS
        DN2 --- MXS
        DNn --- MXS
    end

    %% ===========================
    %% Observability & Stores
    %% ===========================
    subgraph ObservabilityGroup
        PM[Prometheus]
        GF[Grafana]
        AL[Alerting]
        OTEL[OpenTelemetry Collector]
        PM --> GF
        PM --> AL
        OTEL -. traces .- GF
    end

    subgraph Artifacts
        MS[(Model/Artifact Store)]
        VR[Version Registry\nversion-config.json]
    end

    subgraph CICD
        BLD[Build Scripts\nscripts/*]
        PBL[Publish/Release\nrelease.ps1/.sh]
        DKR[Docker Images\ndocker/*]
        K8S[Kubernetes Manifests\nkubernetes/*]
    end

    %% ===========================
    %% Flows
    %% ===========================
    A1 & A2 & A3 & A4 & A5 -->|Text| E1
    E1 -->|Embeddings| E4
    E4 -->|HTTPS + JWT| TLS
    TLS --> Auth --> RL --> CB --> C_API

    C_RT -->|/decode| DN1 & DN2 & DNn
    DN1 -->|Translation| C_API
    C_API -->|Result| A1 & A2 & A3 & A4 & A5

    %% Observability flows
    C_MX --> PM
    MXS --> PM
    C_TR --> OTEL

    %% Model & version flows
    MLD -. downloads .-> MS
    VR -. governs .-> C_CFG

    %% CI/CD & Deploy
    BLD --> DKR --> K8S
    K8S --> Coordinator
    K8S --> DecoderPool
```

---

## Components (In Depth)

### 1) SDKs and Edge Encoder
- **Platforms:** Android, iOS, Flutter, React Native, Web (WASM optional).
- **Encoder Core:** C++ implementation for native platforms with FFI bindings; TypeScript + WASM for the web.
- **Vocabulary System:**
  - Packs of 2–4MB per language (Latin ~3MB, CJK ~4MB, etc.).
  - Dynamic download on demand; cached locally with versioning; LRU for memory efficiency.
- **Output Payload:** Embeddings serialized via MsgPack and compressed with LZ4 to minimize bandwidth.
- **Fallbacks:** SDKs can fall back to cloud‑only encoding if local edge encoding is unavailable.

### 2) Coordinator (Control Plane)
- **Routing:** Least‑loaded healthy node selection with active and passive health probes.
- **Security:** JWT auth for protected endpoints and admin operations; optional HTTPS offload in front of the coordinator.
- **Reliability:** Rate limiter and circuit breaker patterns to protect downstream decoders.
- **Observability:** Exposes Prometheus metrics; integrates with OpenTelemetry for tracing.
- **Dashboard:** Real‑time charts (Chart.js) for node health, throughput, and error rates.
- **Redis (optional):** Shared state for distributed coordination or rate limiting.

### 3) Decoder Nodes (Data Plane)
- **Serving:** Litserve (faster than FastAPI for high‑throughput ML inference) hosting PyTorch models.
- **Model Architecture:** 6‑layer transformer with cross‑attention; supports dynamic adapter loading per language/domain.
- **GPU Acceleration:** Targets GPUs like T4, 3090, V100, A100; memory‑optimized for multi‑concurrency.
- **Health & Metrics:** Each node publishes `/health` and `/metrics` endpoints; autoscaling driven via metrics in K8s.

### 4) Artifacts & Models
- **Locations:** `models/` for local dev; production models fetched by nodes from a model/artifact store.
- **Versioning:** `version-config.json` and scripts coordinate version pins and rollouts.
- **Publishing:** `scripts/upload_artifacts.py` and CI pipelines push artifacts to remote storage.

### 5) Observability
- **Metrics:** Prometheus scrapes coordinator and decoder nodes; dashboards live in `monitoring/grafana/dashboards`.
- **Tracing:** Optional OpenTelemetry integration for end‑to‑end latency analysis.
- **System Metrics:** GPU/CPU/Memory via `GPUtil`, `psutil`; overall health tracked and alerting configured.

### 6) CI/CD & Infrastructure
- **Build:** `scripts/build_models.py`, `scripts/pipeline.py`, and platform‑specific build scripts.
- **Docker:** Dockerfiles under `docker/`; multi‑stage builds for slim images.
- **Kubernetes:** Manifests in `kubernetes/` define services, deployments, probes, and autoscaling policy.
- **Release:** `scripts/release.ps1/.sh` and CI workflows publish SDKs and backend services.

---

## Data Flow (Detailed)

1. The client loads the necessary vocabulary pack(s) if not cached (managed by SDK). 
2. Text is encoded on device via the universal encoder to language‑agnostic embeddings. 
3. Embeddings are serialized (MsgPack) and compressed (LZ4) into a compact payload. 
4. The client calls the coordinator `/decode` over HTTPS with JWT/API key if required. 
5. Coordinator authenticates, rate‑limits, and applies circuit breaker checks. 
6. The router selects the least‑loaded healthy decoder node based on recent metrics and probes. 
7. The selected decoder runs the model to produce the translation. 
8. The decoder returns the translation to the coordinator, which relays it back to the client. 
9. Metrics and traces for the request are exported to Prometheus/OpenTelemetry.

---

## Deployment & Scaling

- **Horizontal Scaling:** Increase the number of decoder nodes; coordinator remains stateless and horizontally scalable. 
- **Autoscaling:** Driven by CPU/GPU utilization, in‑flight requests, and latency metrics. 
- **Node Pools:** Separate GPU pools by model size or language groups for efficient placement. 
- **Kubernetes:** Health probes (`/health`) and readiness gates; per‑service `Service` and `Deployment` manifests. 
- **Canary Releases:** Version pinning via `version-config.json` and label‑based routing in K8s.

---

## Security

- **Auth:** JWT for protected endpoints and admin dashboard; token validation at the coordinator. 
- **Transport:** HTTPS termination at ingress or load balancer; secure headers enforced. 
- **Policies:** Rate limiting, circuit breaker, and request size limits to protect the data plane. 
- **Secrets:** Use environment variables and/or K8s Secrets; never commit secrets to the repo.

---

## Configuration & Versioning

- **Config Sources:** `config/`, environment variables, and CLI flags. 
- **Validation:** `scripts/validate_config.py` validates schema and consistency. 
- **Version Control:** `version-config.json` governs artifact and model versions across environments. 
- **Wizard:** `scripts/config_wizard.py` provides an interactive setup experience.

---

## Ports & Endpoints

- **Decoder Node (default):** Port `8000` — `/decode`, `/health`, `/metrics`. 
- **Coordinator (default):** Port `5100` — `/decode`, `/health`, `/metrics` (+ admin endpoints). 
- **Observability:** Prometheus scrapes `/metrics`; Grafana reads from Prometheus; OTEL collector ingests traces.

---

## Failure Handling & Reliability

- **Retries:** Client‑side retries with backoff for transient network failures. 
- **Circuit Breaker:** Temporarily halts routing to failing nodes; gradual recovery after cooldown. 
- **Timeouts:** Sensible timeouts on client and server; streaming or chunked responses optional. 
- **Graceful Draining:** On shutdown or upgrade, nodes stop accepting new requests but finish in‑flight work. 
- **Fallbacks:** SDKs may fall back to cloud‑only encoding when local edge encoding is unavailable.

---

## Repository Mapping

- `encoder/` — Python encoder logic and training helpers. 
- `encoder_core/` — C++ encoder core for native platforms (FFI‑compatible). 
- `cloud_decoder/` — Decoder service using Litserve (GPU‑ready). 
- `universal-decoder-node/` — Standalone decoder node package (CLI + utils). 
- `coordinator/` — Coordinator with routing, health checks, metrics, and dashboard. 
- `monitoring/` — Prometheus, Grafana dashboards, and metrics utilities. 
- `vocabulary/` — Vocabulary creation and management utilities; packs and evolution tools. 
- `web/universal-translation-sdk/` — Web SDK with TypeScript and WASM build. 
- `react-native/UniversalTranslationSDK/` — React Native SDK with native bridges. 
- `flutter/universal_translation_sdk/` — Flutter SDK with FFI for native performance. 
- `kubernetes/` — Deployment manifests for coordinator and decoder nodes. 
- `docker/` — Dockerfiles for building images for the encoder/decoder services. 
- `scripts/` — Build, release, validation, and automation scripts.

---

## Related Documentation

- [DECODER_POOL.md](./DECODER_POOL.md)
- [SDK_INTEGRATION.md](./SDK_INTEGRATION.md)
- [DEPLOYMENT.md](./DEPLOYMENT.md)
- [CI_CD.md](./CI_CD.md)
- [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md)
- [SECURITY_BEST_PRACTICES.md](./SECURITY_BEST_PRACTICES.md)
- [environment-variables.md](./environment-variables.md)

## Notes

- All services expose `/metrics` for Prometheus scraping.
- Embedding payloads are compact and designed for low‑bandwidth, high‑latency networks.
- WASM builds enable true edge encoding in modern browsers with fallback to cloud when unavailable.

---

## Sequence Diagrams

### A) Request Lifecycle (SDK → Coordinator → Decoder → SDK)

```mermaid
sequenceDiagram
    autonumber
    participant SDK as Client SDK
    participant ENC as Edge Encoder
    participant COORD as Coordinator
    participant RT as Router
    participant DN as Decoder Node
    participant PM as Prometheus
    participant OT as OTEL Collector

    SDK->>ENC: Text input
    ENC->>ENC: Encode to embeddings
    ENC->>COORD: POST /decode (LZ4+MsgPack, JWT)
    COORD->>RT: Select least-loaded healthy node
    RT-->>COORD: Node endpoint
    COORD->>DN: Proxy /decode request
    DN->>DN: Run transformer inference
    DN-->>COORD: Translation + metrics
    COORD-->>SDK: Translation result
    COORD-->>PM: Expose /metrics scrape
    DN-->>PM: Expose /metrics scrape
    COORD-->>OT: Export traces
    DN-->>OT: Export traces
```

### B) Health Probing & Autoscaling Loop

```mermaid
sequenceDiagram
    autonumber
    participant COORD as Coordinator
    participant HC as Health Prober
    participant DN as Decoder Node(s)
    participant PM as Prometheus
    participant HPA as K8s HPA/Autoscaler

    COORD->>HC: Schedule probes
    HC->>DN: GET /health
    DN-->>HC: status (healthy/unhealthy)
    HC->>COORD: Update node registry
    COORD-->>PM: Pool metrics (/metrics)
    DN-->>PM: Node metrics (/metrics)
    PM-->>HPA: Metrics (CPU/GPU/latency/QPS)
    HPA->>DN: Scale up/down replicas
    HPA->>COORD: Notify changes (optional)
```

### C) Model Rollout & Version Pinning

```mermaid
sequenceDiagram
    autonumber
    participant DEV as Dev/CI Pipeline
    participant AS as Artifact Store
    participant VR as Version Registry
    participant K8S as Kubernetes
    participant DN as Decoder Node

    DEV->>AS: Upload model artifacts
    DEV->>VR: Update version-config.json
    DEV->>K8S: Apply deployment with new labels/tags
    K8S->>DN: Rolling update (new pods)
    DN->>AS: Download model/adapters on start
    DN->>VR: Read pinned versions
    DN->>DN: Warm model, ready when loaded
    DN-->>K8S: Ready/Healthy
    K8S-->>DEV: Rollout complete
```

---

## Component-Level Diagram (Modules & Interfaces)

```mermaid
classDiagram
    direction LR

    class SDK {
      +encode(text): Embeddings
      +sendDecodeRequest(payload)
      +handleResult(translation)
      -cache: LocalCache
    }

    class EncoderCore {
      +encode(text): Embeddings
      -loadVocabulary(lang)
      -quantizedOps
    }

    class VocabularyManager {
      +getPack(lang): Pack
      +installPack(lang)
      +evict()
      -cachePolicy: LRU
    }

    class CoordinatorAPI {
      +POST /decode
      +GET /health
      +GET /metrics
    }

    class Router {
      +selectNode(): DecoderNode
      -loadStats
      -healthState
    }

    class DecoderNode {
      +decode(embeddings): Translation
      +/health
      +/metrics
      -model: Transformer
      -adapters
    }

    class MetricsExporter {
      +expose()
    }

    class Tracer {
      +recordSpan()
    }

    class RedisStore {
      +get(key)
      +set(key, value)
    }

    class ArtifactStore {
      +download(modelId)
      +listVersions()
    }

    SDK --> EncoderCore : uses
    SDK --> VocabularyManager : loads packs
    SDK --> CoordinatorAPI : calls

    CoordinatorAPI --> Router : delegates
    Router --> DecoderNode : routes

    DecoderNode --> ArtifactStore : downloads
    CoordinatorAPI --> MetricsExporter : exposes
    DecoderNode --> MetricsExporter : exposes

    CoordinatorAPI --> Tracer : traces
    DecoderNode --> Tracer : traces

    Router --> RedisStore : optional state
```

---

## Deployment Diagram (Kubernetes Topology)

```mermaid
flowchart TB
    subgraph Cluster[Kubernetes Cluster]
        subgraph NS[Namespace: uts-prod]
            subgraph Ingress
                IG[HTTPS Ingress]
            end

            subgraph SVCs
                SCoord[Service coordinator]
                SDec[Service decoder]
                SMon[Service prometheus]
            end

            subgraph Deployments
                subgraph Coord
                    CPod1[Pod coord-abc]
                    CPod2[Pod coord-def]
                end
                subgraph DecPool
                    DPod1[Pod dec-001 - GPU]
                    DPod2[Pod dec-002 - GPU]
                    DPodN[Pod dec-N - GPU]
                end
                subgraph Mon
                    PPod[Pod prometheus]
                    GPod[Pod grafana]
                    OTel[Pod otel-collector]
                end
            end
        end
    end

    IG --> SCoord
    SCoord --> CPod1
    SCoord --> CPod2

    CPod1 --> SDec
    CPod2 --> SDec

    SDec --> DPod1
    SDec --> DPod2
    SDec --> DPodN

    CPod1 --> SMon
    CPod2 --> SMon
    DPod1 --> SMon
    DPod2 --> SMon
    DPodN --> SMon

    SMon --> PPod
    PPod --> GPod
    PPod --> OTel
```