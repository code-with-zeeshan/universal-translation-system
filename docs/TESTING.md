# Testing Guide

Test suite for the Universal Translation System — 42 test modules covering core components.

## Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=. tests/

# Verbose output
pytest tests/ -v

# Specific test file
pytest tests/test_encoder.py
pytest tests/test_pipeline_state.py

# Specific test class or function
pytest tests/test_decoder.py::TestDecoder
pytest tests/test_encoder.py::test_encoder_forward

# Parallel execution
pytest tests/ -n auto
```

## Test Categories

### Encoder (`test_encoder.py`, `test_encoder_quantizer.py`)
- Encoder forward pass, shape checks
- Encoder quantization (INT8, dynamic range)
- Masking, positional encoding

### Decoder (`test_decoder.py`, `test_decoder_smoke_perf.py`, `test_decoder_rs256_auth.py`)
- Decoder forward pass, autoregressive decoding
- RS256/JWT authentication on decoder endpoints
- Throughput and latency benchmarks

### Configuration (`test_config_models.py`, `test_system_config.py`)
- Config model validation (Pydantic schemas)
- System configuration loading and defaults
- Config YAML parsing

### Security (`test_security.py`, `test_sensitive_filter.py`, `test_revocation.py`, `test_credential_manager.py`, `test_secrets_bootstrap.py`, `test_secure_serialization.py`)
- JWT creation/validation
- API key management
- Token revocation
- Secret bootstrap from environment
- Secure pickle/serialization
- Sensitive data filtering in logs

### Training (`test_training_analytics.py`, `test_training_utils.py`, `test_training_strategy.py`)
- Training analytics tracking
- Training utility functions (LR scheduling, checkpointing)
- Training strategy configuration

### Monitoring (`test_monitoring.py`, `test_metrics.py`)
- Prometheus metrics emission
- System health monitoring
- Metric collection and aggregation

### Pipeline (`test_pipeline_state.py`)
- Pipeline checkpoint save/load
- Auto-resume logic
- Config-hash invalidation

### Vocabulary (`test_vocab_config.py`, `test_vocabulary_lru_and_fallback.py`, `test_vocabulary_path.py`)
- Vocabulary pack loading
- LRU cache behavior
- Fallback chain (specific → script group → default)
- Path resolution

### API (`test_translation_api.py`)
- `/decode` endpoint behavior
- Header validation
- Error response codes

### Coordinator (`test_coordinator_rs256_validation.py`)
- RS256 token validation in coordinator
- Pool membership authentication

### Integration (`test_complete_integration.py`, `test_integration_fixes.py`)
- End-to-end encoder → decoder pipeline
- Cross-component integration

### System (`test_system_health.py`, `test_env_hygiene.py`, `test_constants.py`)
- Health check endpoints
- Environment variable hygiene
- Constant correctness

### Quality (`test_quality_comparator.py`)
- BLEU, chrF, COMET computation
- Quality comparison utilities

### Utilities (`test_artifact_store.py`, `test_custom_samplers.py`, `test_hardware_profile.py`, `test_hardware_compile_recommendation.py`, `test_logging_filter.py`, `test_rate_limiter.py`, `test_thread_safety.py`, `test_version_compat.py`, `test_health_readiness_endpoints.py`)
- Artifact storage backend
- Custom data samplers
- Hardware profiling and compilation recommendations
- Logging filters
- Rate limiter
- Thread safety utilities
- Version compatibility checks
- Health/readiness probe endpoints

## Test Runner (`run_tests.py`)

A standalone test runner that replicates conftest behavior for non-pytest environments. Runs a subset of critical tests:

```bash
python run_tests.py
```

Subset includes: `test_encoder_quantizer`, `test_quality_comparator`, `test_hardware_profile`, `test_training_analytics`, `test_training_utils`, `test_system_health`, `test_translation_api`.

## Adding Tests

1. Create `tests/test_<feature>.py`
2. Import from `unittest` (preferred) or use pytest conventions
3. Use `tests/conftest.py` fixtures for shared setup
4. Run `pytest tests/test_<feature>.py -v` to verify

### Test Fixtures (conftest.py)

The conftest provides:
- Mock encoder and decoder model instances
- Config fixtures with test defaults
- Temporary directories for pipeline state
- KV cache fixtures for decoder testing
