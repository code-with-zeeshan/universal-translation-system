# FAQ: Universal Translation System

## Core Differentiation Questions

### Q1: Why not just use M2M-100 or NLLB-200 quantized?
Our system uses an innovative edge-cloud split architecture with a universal encoder (35MB base + 2-4MB vocabulary packs) and cloud decoder infrastructure. This results in a 40MB app with 90% quality of full models. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Q2: What makes your vocabulary pack system unique?
Vocabulary packs are small (2-4MB each), dynamically loaded, and language-specific. Users download only the languages they need. See [docs/Vocabulary_Guide.md](docs/Vocabulary_Guide.md).

### Q3: How do you maintain quality with such a small model?
Through smart quantization, optimized vocabulary packs, and edge-cloud split architecture. Heavy lifting is done on the cloud decoder. See [docs/VISION.md](docs/VISION.md).

## Technical Architecture Questions

### Q4: Why split encoder and decoder?
Minimizes client app size while maximizing translation quality. Encoder runs on device, decoder in the cloud. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Q5: How is privacy preserved?
Only embeddings are sent to the cloud; original text never leaves the device. Embeddings are compressed and cannot be reversed.

### Q6: What about offline translation?
We're working on a fully offline mode for limited connectivity scenarios.

## Business/User Questions

### Q7: Who is this system designed for?
Developers, privacy-conscious users, and organizations needing scalable translation. SDKs support Android, iOS, Flutter, React Native, and Web (under `sdk/`). See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md).

### Q8: How does configuration work?
All components configurable via environment variables. Paths overridable via `UTS_*` env vars. See [docs/environment-variables.md](docs/environment-variables.md).

### Q9: What languages are supported?
20 languages (en, es, fr, de, zh, ja, ko, ar, hi, ru, pt, it, tr, th, vi, pl, uk, nl, id, sv) with plans to expand.

## Developer Questions

### Q10: How hard is it to integrate?
SDKs designed for easy integration with minimal code. See [docs/SDK_INTEGRATION.md](docs/SDK_INTEGRATION.md) for examples.

### Q11: Can I use my own vocabulary/terminology?
Yes, create custom vocabulary packs. See [docs/Vocabulary_Guide.md](docs/Vocabulary_Guide.md).

### Q12: How does deployment work?
Docker Compose, standalone Docker, Helm chart, and Kubernetes manifests. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Role-based install via `scripts/install.sh`.

## Future/Roadmap Questions

### Q13: What's the roadmap?
See [docs/future_plan.md](docs/future_plan.md) for details on offline capabilities, more languages, and enhanced monitoring.

### Q14: How do you handle model updates?
Through environment variable configuration system and `version-config.json` for component semver management.

### Q15: What makes this system unique?
Edge-cloud split architecture, dynamic vocabulary system, environment variable configuration, centralized path management, thread-safe design, and production-ready deployment scripts.

---

For more information, see the documentation in the `/docs` folder and explore the coordinator dashboard for real-time system status.
