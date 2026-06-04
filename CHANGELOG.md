# Changelog

All notable changes to the Universal Translation System will be documented in this file.

## [Unreleased]

### Added
- TUI dashboard: `tui/` package with pipeline, training, GPU, and log panels
- `uts` unified CLI (`scripts/uts.py`) — single entry point organizing all tools by workflow
- `./uts` shell entry point at project root
- `--eval-only` flag for data pipeline
- `vocab_size` field in `VocabularyConfig` schema (configurable from YAML)
- `--vocab-size` CLI flag for vocabulary rebuild
- Architecture doc with dual-phase strategy (full training → LoRA adaptation)

### Changed
- **Config: Full model training** — `use_lora: false`, `num_epochs: 10`, `lr: 3e-4`, `warmup_steps: 1000`
- **Vocabulary size**: 32K tokens per pack (was 25K), matching embedding table
- `UnifiedVocabConfig.vocab_size` default: 25000 → 32000
- `VocabularyConnector.create_vocabularies_from_pipeline()` accepts `vocab_size` param
- README.md, SETUP_COMMANDS.md, ONBOARDING.md, TRAINING.md, ARCHITECTURE.md rewritten
- FAQ.md, TROUBLESHOOT.md updated with current state
- Pipeline defaults skip `wikipedia_backtranslation`, `direct_opus`, `knowledge_distillation` (opt-in via `--stage`)
- `ut` language code → `id` for Indonesian in multiple files
- `frequency` → `frequency` dict key in `vocab_production.py` 

### Fixed
- Vocabulary import path bugs, logging TypeErrors, auto-reduce `vocab_size` on RuntimeError
- False friend generation: closed file bug, batching all translations before NLLB call
- `vocabulary_creator.py` evolution save bug: `_create_pack_structure` + `_save_pack` no longer short-circuits
- `vocabulary_monitor.py`: `from metrics_collector` → `from monitoring.metrics_collector`
- `data/essential/` directory removed (never written by any stage)
- `pathlib2` removed from `requirements/dev.txt` (Python 3.12 stdlib)
- `opentelemetry-instrumentation-flask` removed from `requirements/serve.txt`
- 4 dead `.py` files deleted: `error_codes.py`, `error_handler.py`, `final_integration.py`
- 16 dead non-`.py` files deleted: duplicate configs, orphaned requirements, `.ps1` scripts, redundant `.sh` scripts, deprecated K8s PodSecurityPolicy manifests, `.html` flow diagrams
- Environment variable configuration for all components
- Docker and Kubernetes deployment support
- Comprehensive Prometheus/Grafana monitoring dashboards
- VISION.md document explaining the system architecture and goals
- Enhanced train_from_scratch.py script with improved command-line interface
- Reorganized documentation structure for better navigation
- Comprehensive Prometheus configuration with alerting and recording rules
- Security best practices documentation
- Kubernetes health probes and resource requests
- Improved Docker health checks using wget instead of curl
- SDK_PUBLISHING.md guide (Android Maven, iOS Podspec/SPM, RN linking)
- Web example Express server with proper WASM headers (COOP/COEP/CORS)
- README updates for Android/iOS/Flutter/Web SDKs with coordinator usage
- GitHub Actions workflows: sdk-publish.yml and web-npm-publish.yml
- Coordinator periodic Redis-to-disk mirroring via `COORDINATOR_MIRROR_INTERVAL`
- Centralized logging via `utils.logging_config.setup_logging`
- Automatic creation of logs folder structure via `DirectoryManager.create_logs_structure()`
- Mandatory secrets in `.env.example` with `*_FILE` support and RS256 key envs
- Production scripts: `scripts/install.sh`, `scripts/build_encoder_core.sh`, `scripts/setup_redis.sh`, `scripts/setup_serving.sh`
- Helm chart at `charts/uts/` with coordinator, decoder, encoder, redis deployments
- Kubernetes secrets template at `kubernetes/secrets.example.yaml`
- Thread safety: 19 race conditions fixed across 11+ files (RLock, background thread stop, double-checked locking)
- 285+ tests across 14 new test files covering constants, samplers, strategy, config, metrics, pipeline state, vocab config, quantizer, profiler, hardware, analytics, training utils, health, translation API
- 60+ path constants in `utils/constants.py` with `UTS_*` env-var overrides
- `version-config.json` + `scripts/version_manager.py` for component semver management
- Role-based install via `scripts/install.sh` (`--train`, `--serve`, `--coordinator`, `--dev`, `--encoder-core`, `--all`)
- Pinned Docker images (`prom/prometheus:v2.51.0`, `grafana/grafana:10.4.0`, `ubuntu:22.04`)
- Missing packages added to requirements: `nvidia-ml-py3`, `semver`, `urllib3` (base); `seaborn` (train); `tf2onnx`, `torch-tensorrt`, `tritonclient` (export)

### Changed
- SDKs moved from root directories to `sdk/` subdirectory; ~70 references updated across workflows, scripts, docs, configs
- `config/config_models.py` merged into `config/schemas.py`; canonical hierarchy with `load_system_config()` for serving stack
- `SystemConfig` renamed to `IntegrationSystemConfig` in `integration/system_config.py` to avoid name collision
- `IntelligentTrainer` inherits from `BaseTrainer` eliminating unused abstract class
- `TemperatureSampler` consolidated from 3 copies to canonical `data/custom_samplers.py`
- Fake quantization merged into shared `fake_quantize_tensor()` in `training/quantization_common.py`
- Gradient checkpointing harmonized to model built-in API
- Oversized files split into 30 modules + 7 shims (backward-compatible re-exports):
  - `training/intelligent_trainer.py` (1,856 lines) → `trainer.py`, `hardware_profile.py`, `training_analytics.py`, `training_strategy.py`
  - `integration/connect_all_systems.py` (995) → `system.py`, `system_config.py`, `system_health.py`, `translation_api.py`
  - `vocabulary/unified_vocabulary_creator.py` (1,059) → `vocabulary_creator.py`, `vocab_production.py`, `vocab_research.py`, `vocab_validation.py`, `vocab_config.py`
  - `evaluation/evaluate_model.py` (781) → `evaluator.py`, `metrics.py`
  - `training/quantization_pipeline.py` (844) → `encoder_quantizer.py`, `model_profiler.py`, `quality_comparator.py`, `quantization_common.py`
  - `training/memory_efficient_training.py` (784) → `memory_trainer.py`, `memory_tracker.py`, `dynamic_batch_sizer.py`, `memory_config.py`
  - `data/unified_data_pipeline.py` (765) → `pipeline_orchestrator.py`, `pipeline_state.py`
- Import hygiene: 35 double imports fixed, 2 wildcard imports removed, 14 `from __future__ import annotations` removed
- Docker/K8s fixes: coordinator healthcheck port corrected (`8002→5100`), Helm ports updated (`8080/9000→5100/8001`), `litserve→uvicorn` in decoder.Dockerfile, `configs/→config/` path typo fixed
- Decoder Dockerfile: corrected from litserve to uvicorn command
- Coordinator healthcheck port aligned with actual service port
- Docker images pinned to specific versions instead of `:latest`
- Non-root user (`adduser`) added to all Dockerfiles
- `SensitiveDataFilter` renamed to `LoggingSensitiveDataFilter` in `utils/logging_config.py`
- `track_translation_request` consolidated to re-export from `monitoring/metrics.py`
- Test consolidation: merged `test_coordinator_ready.py` + `test_probes_and_routers.py` + `test_decoder_endpoints.py` → `test_health_readiness_endpoints.py`
- `sha256_file()` + `find_schema_files()` extracted into `scripts/_shared.py`
- Syntax error `status='error''` fixed in `monitoring/metrics_collector.py:135`
- 10 custom modules replaced with library equivalents, eliminating 4,251 lines of custom code
- 5 zero-usage modules deleted (cache_manager, dependency_container, batch_processor, lazy_loader, service_discovery)
- All 5 `archived/` directories deleted (16+ files, ~5,000 lines)
- 8 blocking GPU pipeline bugs fixed

### Security
- Decoder enforces `DECODER_JWT_SECRET` or file at startup (fail-fast)
- Coordinator supports `COORDINATOR_SECRET(_FILE)`, `COORDINATOR_JWT_SECRET(_FILE)`, `COORDINATOR_TOKEN(_FILE)`, `INTERNAL_SERVICE_TOKEN(_FILE)`
- RS256 support documented: `JWT_PRIVATE_KEY_FILE`, `JWT_PUBLIC_KEY_PATH`; Kubernetes placeholders included
- Thread safety: 19 race conditions fixed across 11+ files

## [0.1.0] - 2025-08-22

### Added
- Initial release of Universal Translation System
- Edge encoding, cloud decoding architecture
- Support for 20 languages with dynamic vocabulary loading
- Native SDKs for Android, iOS, Flutter, React Native, and Web
- Basic monitoring with Prometheus metrics
- Coordinator for load balancing and health monitoring
