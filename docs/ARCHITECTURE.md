# System Architecture

## Overview

The Universal Translation System uses a split architecture to minimize client size while maximizing translation quality and performance.

## Components

### 1. Universal Encoder (Edge/Client)
- **Platforms:** Android, iOS, Flutter, React Native, Web
- **Implementation:**
  - Android/iOS/Flutter: Native C++ core via FFI (libuniversal_encoder)
  - React Native/Web: API-based encoding (optionally native in future)
- **Vocabulary:** Dynamic loading (2-4MB per language pack)
- **Output:** Compressed embeddings (2-3KB per translation)

### 2. Vocabulary Packs
- **Latin Pack:** ~3MB (covers 12 languages)
- **CJK Pack:** ~4MB (Chinese, Japanese, Korean)
- **Other Packs:** 1-2MB each

### 3. Universal Decoder (Cloud)
- **Implementation:** PyTorch, served via Litserve (2x faster than FastAPI)
- **Architecture:** 6-layer transformer with cross-attention
- **Infrastructure:** Runs on GPU servers (T4, 3090, V100, A100)
- **Deployment:** Docker, Kubernetes, supports horizontal/vertical scaling

### 4. Advanced Coordinator
- **Role:** Manages communication between multiple edge encoders and a dynamic pool of cloud decoders
- **Features:**
  - Least-loaded load balancing (routes requests to the decoder with the lowest current load)
  - Dynamic decoder pool: add/remove decoders at runtime via REST API or dashboard, no downtime
  - Health checks: background thread checks each decoder’s `/health` endpoint
  - Prometheus metrics: exposes coordinator and decoder pool metrics for monitoring
  - Authentication: token-based for admin endpoints and dashboard actions
  - Web UI: dashboard for monitoring, manual routing, and node management
  - **Enhanced Dashboard:**
    - Authentication UI for admin actions
    - Real-time charts (load, uptime) using Chart.js
    - Advanced analytics: uptime, request rates, error rates, per-decoder stats
    - Manual routing for authenticated users
- **How it works:**
  - Encoders send requests to the coordinator’s `/decode` endpoint
  - Coordinator selects the least-loaded healthy decoder and proxies the request
  - Decoders can be added/removed at any time; the system automatically adjusts
  - All activity and health is visible in the dashboard and via Prometheus

## Data Flow

1. User inputs text
2. App loads relevant vocabulary pack (if needed)
3. Encoder converts text → embeddings (on device or via API)
4. Embeddings compressed and sent to coordinator `/decode` endpoint
5. Coordinator load-balances and proxies to a healthy decoder
6. Decoder generates translation (Litserve endpoint)
7. Translation sent back to app

## SDK Alignment Table

| SDK           | Edge Encoding | Cloud Decoding | Native/FFI | API-based | Aligned? |
|---------------|--------------|---------------|------------|-----------|----------|
| Android/iOS   | Yes          | Yes           | Yes        | Yes       | Yes      |
| Flutter       | Yes          | Yes           | Yes        | Yes       | Yes      |
| React Native  | No           | Yes           | No         | Yes       | Yes      |
| Web           | No           | Yes           | No         | Yes       | Yes      |

## Key Design Decisions

- **Split Architecture:** Small client, heavy compute on server
- **Universal Encoder:** One encoder, dynamic vocab, zero-shot
- **Litserve for Inference:** Fast, production-grade AI serving
- **Advanced Coordinator:** Scalable, dynamic, observable, and secure routing for all decoders
- **CI/CD:** Automated builds for encoder/decoder, artifact storage
- **Config Auto-Detection:** Training scripts auto-select best config for detected GPU

## Directory Structure
(see project root for up-to-date structure)