# Changelog

All notable changes to the Universal Translation System will be documented in this file.

## [Unreleased]

### Added
- **Language ID token** (`runtime/encoder/universal_encoder.py`): `nn.Embedding(20, hidden_dim)` prepends a per-language bias to all input token positions via `language` parameter in `forward()`. List of 20 supported languages (`en, es, fr, de, zh, ja, ko, ar, hi, ru, pt, it, tr, th, pl, uk, nl, id, sv, vi`). ONNX export updated with `language` input alongside `input_ids`/`attention_mask`.
- **Early BLEU validation** (`pipeline/training/trainer.py`): `_evaluate_bleu()` method generates on a subset of the validation set and computes corpus BLEU using `sacrebleu`. Runs after epoch 1 in the training loop; score logged and stored in `training_history['bleu_scores']`.
- **Auto-evolve vocab for low-resource pairs** (`pipeline/data/orchestrator.py`): After `_sample_and_filter_data`, pairs with <50% of `target_size` trigger `VocabularyEvolver.evolve_all_packs()` to expand the relevant script-group vocab packs with new subwords.
- **Pipeline stage categories** (`scripts/_wizard_shared.py`, `scripts/uts.py`, `config/base.yaml`): Every stage tagged with GPU profile (`cpu`/`gpu_light`/`gpu_heavy`). TUI stage selector groups by category with color-coded headers and realistic time estimates. New `--gpu-level` flag on `uts data --pipeline` filters stages by GPU capability (`cpu`=CPU-only, `gpu_light`=CPU+light GPU, `gpu_heavy`=all). `base.yaml` includes `stage_profiles` section for reference.
- **HF Hub data sync** (`pipeline/data/hub_sync.py`): `upload_processed_data()` and `download_processed_data()` for syncing train/val data + vocab packs to/from HF Hub dataset repos. Auto-upload after data pipeline completes if `hub.auto_upload: true`; auto-download before training if data missing locally and `hub.auto_download: true`.
- **Dual HF repo support** — `HubConfig.dataset_repo_id` (defaults to `code-with-zeeshan/UTS-Datasets`) for data+vocab sync, `HubConfig.model_repo_id` (defaults to `code-with-zeeshan/Universal-Translation-System`) for model publish. `hub:` section added to `config/base.yaml` with both defaults and `auto_upload`/`auto_download` flags.
- **`--hub-repo-id` CLI flag** on `uts data --pipeline` and `uts train --full` (overrides config `hub.dataset_repo_id`)
- **`--no-hub-download` CLI flag** on `uts train --full` (disables auto-download for offline/disconnected runs)
- **Interactive config builder** (`uts config` subcommand in `scripts/uts.py`, `scripts/config_interactive.py`): TUI with stage selector (time estimates), training/model/data overrides, review & save. `uts config --set key=value` for batch, `uts config --list`/`--diff` for management. Saves complete merged YAML to `config/override/<name>.yaml`.
- **`_wizard_shared.py`** (`scripts/_wizard_shared.py`): Single source of truth for stage definitions (names, descriptions, time estimates, group membership, default states). Shared by both `config_interactive.py` and `data_pipeline_wizard.py`, eliminating code duplication.
- **HF Hub card files** at `/home/user/hf_upload/`: `model_readme.md` + `model_gitattributes` for model repo, `dataset_readme.md` + `dataset_gitattributes` for dataset repo.

### Changed
- **Encoder/decoder dimension unified to 512** (`config/base.yaml`): `decoder_dim: 768→512`, `decoder_heads: 12→8` (head_dim stays 64). All runtime defaults (`decoder_core.py`, `decoder_server.py`, `udn/decoder.py`, `udn/config.py`) updated from `encoder_dim=1024`/`decoder_dim=512` to both 512. Bootstrap defaults (`bootstrap.py`, `scripts/pipeline.py`) updated. `encoder_adapter` becomes identity `512→512` — fewer params, no quality loss.
- **`scripts/data_pipeline_wizard.py`**: Refactored from standalone TUI implementation to thin CLI wrapper that imports stage definitions from `_wizard_shared.py`.
- **`scripts/config_interactive.py`**: Stage definitions sourced from `_wizard_shared.py` instead of local tuple (was drifting from data pipeline wizard).
- **`docs/ONBOARDING.md`**: Added `uts config` section between vocab and train workflow steps; marked `setup --config-wizard` as legacy; added `--hub-repo-id` flags to data/train command tables.
- **`docs/RUNTIME_LAYOUT.md`**: Added `config/override/` directory to project layout; listed `config_interactive.py` in "Created by" section.
- **`README.md`**: Added `uts config` row to workflow table (1 config topic); CLI topics count updated from 28→33.
- **`pipeline/training/launch.py`**: `load_datasets()` checks `hub.dataset_repo_id` + `hub.auto_download` before failing with "data not found" — auto-downloads from HF if configured.
- **`config/base.yaml`**: Added `hub:` section with `dataset_repo_id: code-with-zeeshan/UTS-Datasets`, `model_repo_id: code-with-zeeshan/Universal-Translation-System`, `token: ""`, `auto_upload: false`, `auto_download: true`.

### Fixed
- **C1 — Data config erased on load**: `config/schemas.py:441` — `config_data['data'] = {}` overwrote all YAML data config (languages, training_distribution, augmentation_pairs). Changed to `config_data.setdefault('data', {})`.
- **C3 — Batch probe crashes on GPU**: `pipeline/training/trainer.py:745` — `safe_sizes.index(found)` crashed when batch probe returned non-standard size (most GPUs). Changed to `max(i for i, s in enumerate(safe_sizes) if s <= found)`.
- **C6 — String literal defaults in pipeline CLI**: `scripts/pipeline.py:306-307` — `--encoder-out`/`--decoder-out` defaults were literal Python expression strings, not evaluated paths. Changed to `default=None` with path resolution at call time.
- **H1 — RuntimeDirectoryManager ignores config**: `pipeline/training/trainer.py:109` — `RuntimeDirectoryManager()` created with default `root="output"`, ignoring user config. Changed to `RuntimeDirectoryManager(config=config)`.
- **H7 — Eager import crashes training module**: `pipeline/training/trainer.py:60` — `from pipeline.training.samplers import TemperatureSampler` at module top-level. Moved to lazy import inside conditional block.
- **Dead batch-size methods**: `pipeline/training/trainer.py:1289,1303` — `decrease_batch_size()`/`increase_batch_size()` called non-existent methods on `DynamicBatchSizer` (runtime crash). Changed to `adjust_batch_size(delta=-1)`/`adjust_batch_size(delta=1)`.

### Changed
- **All runtime paths now managed by `RuntimeDirectoryManager`** — 16 files with hardcoded paths + 20 files importing path constants from `utils.constants` migrated to RDM. `utils/constants.py` path constants deprecated in favor of `RuntimeDirectoryManager` (kept for backward compat and env-override)
- **`utils/constants.py` docstring** updated: path constants marked as deprecated, pointing to RDM
- **`utils/logging_config.py`**: 6 section handler filenames changed from `LOG_DIR` constant to `log_dir` parameter (defaults to `RDM().logs_dir`). `LOG_DIR` import removed.
- **`utils/common_utils.py`**: `RuntimeDirectoryManager.config` typed as `Optional[RootConfig]` (was `Optional[object]`). `LOG_DIR` import removed from this file.
- **`utils/logging_config.py`**: `setup_logging()` default `log_dir` changed to `str(RuntimeDirectoryManager().logs_dir)`
- **`utils/logging_config.py`**: `create_logs_structure()` now receives caller-supplied `log_dir` (was hardcoded `RDM().logs_dir`)
- **`utils/logging_config.py`**: root log file renamed from `translation_system.log` → `universal_translation_system.log`
- **`utils/logging_config.py`**: per-component `error.log` files added to all 7 section subdirectories (`data/`, `training/`, `monitoring/`, `coordinator/`, `decoder/`, `vocabulary/`, `evaluation/`) — each component writes ERROR level to both its own `error.log` and the root `errors.log`
- **`utils/logging_config.py`**: `evaluation` namespace now has dedicated logger, handler, and error handler (was orphan — previously wrote only to root)
- **`utils/logging_config.py`**: silent `except Exception: pass` on directory creation replaced with `logger.debug(..., exc_info=True)`
- **Eval data stages** (`download_evaluation`) removed from default `enabled_stages` in `PipelineConfig`. Heavy stages (`comet_quality`, `wikipedia_backtranslation`, `direct_opus`, `knowledge_distillation`) opt-in via config or `uts data --interactive`
- **Decoder consolidation**: `runtime/cloud_decoder/decoder_server.py` now imports model classes from `decoder_core.py` instead of maintaining its own copy (~140 lines removed)
- **`DataConfig.seed`** now read directly as `self.config.data.seed` (was accessed via `getattr` fallback)
- **`pipeline/data/orchestrator.py::main()`**: `print()` → `logger.info()`
- **`pipeline/data/downloader.py::main()`**: `print()` → `logger.info()`
- **`runtime/cloud_decoder/optimized_decoder.py`**: Last `print()` → `logger.info()` in config reload handler
- **`runtime/coordinator/advanced_coordinator.py`**: All `os.path.*` calls (11) replaced with `pathlib.Path` equivalents
- **`pipeline/training/launch.py`**, **`monitoring/metrics_collector.py`**: `sys.path` hacks removed (kept in standalone scripts only)

### Fixed
- **8 CLI dispatch bugs**: (1) `uts.py:166` — undefined `config` → `config_path`; (2-4) `uts.py:356,375,377` — wrong module paths `cloud_decoder/` → `runtime/cloud_decoder/`, `coordinator/` → `runtime/coordinator/`; (5) `uts.py:170` — `--domains` → `--domain` arg name mismatch; (6) `uts.py:249` — non-distributed train called `trainer` (no CLI), now always dispatches to `launch`; (7) `scripts/compatibility_checks.py:150` — missing `import hashlib`; (8) `scripts/pipeline.py:124,264` — string literal paths now resolved via `RuntimeDirectoryManager()`
- **3 critical bugs**: string literal defaults in `pipeline/training/datasets.py` and `scripts/pipeline.py` (vocab_dir, encoder-out); duplicate import in `pipeline/data/orchestrator.py`
- **All `except Exception: pass` blocks** — 7 blocks across `evaluation/evaluator.py`, `runtime/vocabulary/manager.py`, `integration/system_health.py` (x2), `runtime/cloud_decoder/optimized_decoder.py` (x3): changed to `logger.*warning|debug*(..., exc_info=True)`. No bare `except:` remains in production code.
- **Unused imports** — 16 removed: `litserve`, `msgpack`, `yaml`, `gpu_utilization` from `optimized_decoder.py`; `torch.nn.functional`, `FSDP`, `transformer_auto_wrap_policy`, `psutil`, `warnings`, `dataclass` from `memory/trainer.py`; 6 unused re-exports from `utils/__init__.py`
- **Dead code** — unreachable block in `pipeline/training/trainer.py:495-503` removed; `utils/dataset_classes.py` stub deleted (4 imports updated); `vocabulary/` top-level shim deleted
- **Duplicate imports**: `from pathlib import Path` duplicated in `downloader.py`; `import sys` unused in `downloader.py::main()`; 2 local `from pathlib import Path` inside functions in `advanced_coordinator.py`
- **Auto-resume pipeline** (`utils/pipeline_checkpoint.py`): `PhaseCheckpoint` class with config-hash fingerprinting, cross-stage data→train→eval tracking via global `pipeline_state.json`, sub-stage per-pair completion tracking, and `invalidate_downstream()` for config-change detection
- **Per-pair checkpointing** for data pipeline sub-stages: idiom, false-friend, dynamic, and backtranslation files tracked individually so partial runs skip only what's done
- **Dynamic batch sizing for NLLB pipeline** (`data/synthetic_augmentation.py`): `_probe_pipeline_batch_size()` runs progressive `model.generate()` from 16→probe_limit at startup, finding the true max batch size for the GPU (L4 finds ~1024, A100 80GB finds ~2048)
- **Dynamic batch sizing for training** (`training/dynamic_batch_sizer.py`): `probe()` now scales probe limit with GPU memory (`total_gb * 10`), allowing H100/A100 80GB to find batch sizes beyond the old 256 cap
- **CLI flags for download tuning**: `--download-max-workers`, `--download-parallel-batches`, `--datasets-cache-dir` on `uts data --pipeline` and `uts data --download-only`
- **`--force` flags** on `uts data`, `uts train`, `uts eval` — re-run any stage from scratch, invalidating downstream
- **Knowledge distillation** (`pipeline/training/distillation.py`): KL-divergence loss from NLLB-200-3.3B teacher, configurable alpha/temperature, `uts train --distill`
- **Evaluation per-file checkpointing**: tracks individual test file completion, auto-resumes mid-eval
- **Coordinator batcher** (`runtime/coordinator/advanced_coordinator.py`): 50ms accumulation window per endpoint, concurrent forward to LitServe
- **Coordinator `/api/status` endpoint**: returns `single_decoder: bool` — SDKs use this for routing decisions
- **mDNS broadcast**: decoders advertise as `_universal-translate._tcp.local.`, SDKs auto-scan localhost ports
- **Local decoder GPU detection** (`universal-decoder-node/udn/cli.py`): detects CUDA/MPS, prompts user, saves to `decoder_config.yaml`
- **Circuit breaker** integrated in coordinator (CLOSED/OPEN/HALF-OPEN states)
- All 5 SDKs: coordinator-aware routing, local decoder preference, port auto-scan, configurable `hfRepo`
- Flutter SDK: `translate()` uses `sourceLang`/`targetLang` (was `from`/`to`)
- Android/iOS SDKs: `localDecoderUrl` + `checkLocalDecoder()` with fallback scan `[8000,8080,9000]`
- TUI dashboard: `tui/` package with pipeline, training, GPU, and log panels
- `uts` unified CLI (`scripts/uts.py`) — single entry point organizing all tools by workflow
- `./uts` shell entry point at project root
- `--eval-only` flag for data pipeline
- `vocab_size` field in `VocabularyConfig` schema (configurable from YAML)
- `--vocab-size` CLI flag for vocabulary rebuild
- Architecture doc with dual-phase strategy (full training → LoRA adaptation)
- `uts publish` group: split checkpoint → ONNX export → quantize → HF Hub upload, with `--preflight` and `--optimize-decoder`
- `uts tui` group: terminal UI dashboard for pipeline/training/GPU monitoring
- `uts tools --register-decoder`, `--build-encoder`, `--check-compat`, `--set-env`, `--key-type`
- `uts setup --verify`: post-deployment verification
- `uts data --domains`, `--scale`, `--no-resume`, `--force`
- `uts train --distill`, `--force`, `--start-tier`, `--validate-final`
- `uts eval --force`
- `docs/TUI.md`, `docs/PUBLISHING.md`, `docs/TESTING.md`, `docs/RUNTIME_LAYOUT.md`, `docs/VERSION_MANAGEMENT.md`, `docs/SECRET_MANAGEMENT.md`
- Wired 9 tools into `uts`: `setup --verify`, `data --domains`, `train --distill`, `publish --preflight/--optimize-decoder`, `tools register-decoder/build-encoder/check-compat`

### Changed
- **Default epochs: 10 → 5** (`config/base.yaml`), budget-friendly ($4.65 vs $9.30 on A100)
- **Config-hash invalidation**: config change on any stage auto-invalidates that stage + all downstream
- **Eval data downloads on demand** during `uts eval --model`, not during data pipeline
- **Package renamed**: `universal_decoder_node` → `udn`, CLI command is `udn`
- **HF Hub layout**: `models/production/encoder.pt, decoder.pt, encoder.onnx` → `models/production/`; `vocabulary/vocab/*.msgpack` → `vocabs/`; `models/adapters/*.pt` → `adapters/`
- **Coordinator**: single decoder → SDK calls directly; multiple decoders → proxy through coordinator batcher
- **`.env.example`**: added `UTS_ROLE`, `UTS_VOCAB_SIGNING_KEY`, `UTS_JWT_DEFAULT_ALG`, `JWT_SECRET`, `DECODER_CONFIG`, `EVOLVE_ANALYTICS_JSON`; removed dead `ENCODER_HOST`
- `docs/environment-variables.md` rewritten from 54 vars to 150+ var complete reference
- `README.md`: CLI groups (8→10), training timings (6h→3h budget), component status refreshed, doc section expanded, features list updated
- `universal-decoder-node/__init__.py`: `__version__` 0.1.0 → 1.0.0 (was mismatched with pyproject.toml/setup.py/version-config.json)
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
- **Audit fixes**: C1 (import circular), C4 (phantom decoder stub), W1 (field mapping), W2 (Flutter params), W3 (Android/iOS local decoder), W5 (openapi/proto dead path), W7 (empty dir), W8 (docstring), W9 (3 dead-code files: `sensitive_filter.py`, `validation_decorators.py`, `training_validator.py`), I1 (HF repo configurable), I3 (CLI flag conventions `--repo-id` vs `--repo_id`)
- **Training cross-stage fix**: `is_stage_complete("data", "")` → actual data config hash check
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
- GitHub Actions workflows: build-upload.yml, publish-pypi.yml
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
