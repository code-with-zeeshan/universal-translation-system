# Performance Optimization

This document describes the performance optimization features of the Universal Translation System, including memory management, profiling, and bottleneck detection.

## Memory Management

The Universal Translation System includes a comprehensive memory management system via `utils/resource_monitor.py`, `utils/resource_tracker.py`, and `training/memory_*.py` modules.

### Key Features
- **Real-time Memory Monitoring**: Continuous monitoring of system and GPU memory usage
- **Automatic Cleanup**: Configurable thresholds for triggering memory cleanup
- **Model Optimization**: Automatic optimization of models for inference
- **Alert System**: Configurable alerts for memory usage thresholds
- **Thread Safety**: RLock-protected ResourceMonitor singleton

### Configuration
```yaml
memory:
  enable_monitoring: true
  monitoring_interval_seconds: 60
  memory_threshold_percent: 85
  gpu_memory_threshold_percent: 85
  auto_cleanup: true
  cleanup_threshold_percent: 80
```

### Usage
```python
# Resource monitoring
from utils.resource_monitor import ResourceMonitor
monitor = ResourceMonitor()
stats = monitor.get_memory_stats()

# Memory-optimized training
from training.memory_trainer import MemoryOptimizedTrainer
trainer = MemoryOptimizedTrainer(config, model)
```

### Memory Monitoring Endpoint
The decoder service `/status` endpoint provides memory usage information:
```json
{
  "status": "healthy",
  "uptime": "3h 12m 45s",
  "version": "1.2.0",
  "memory": {
    "system_memory_percent": 65.2,
    "gpu_memory_percent": 72.8,
    "system_memory_used_gb": 12.4,
    "gpu_memory_used_gb": 8.6,
    "system_memory_total_gb": 16.0,
    "gpu_memory_total_gb": 12.0
  }
}
```

## Profiling System

The profiling system lives in the `universal-decoder-node` package and `training/model_profiler.py`.

### Key Features
- **Function Profiling**: Detailed timing of function execution
- **Section Profiling**: Granular profiling of code sections
- **Bottleneck Detection**: Automatic identification of performance bottlenecks
- **Export Capabilities**: Multiple export formats (JSON, CSV, TXT)
- **History Tracking**: Performance trends over time

### Configuration
```yaml
profiling:
  enable_profiling: true
  profile_output_dir: "profiles"
  bottleneck_threshold_ms: 100.0
  export_format: "json"
```

### Usage
```python
from universal_decoder_node.utils.profiler import profile, profile_section, function_profiler

@profile
def my_function():
    pass

with profile_section("data_processing"):
    result = process_complex_data(data)

stats = function_profiler.get_stats()
bottlenecks = function_profiler.identify_bottlenecks()
function_profiler.export_stats(filepath="profiles/my_profile.json")
```

### Training Profiling
```bash
python -m training.launch profile --config config/base.yaml --profile-steps 20 --benchmark
```
See `training/training_analytics.py` and `training/model_profiler.py`.

### Profiling Endpoints
- `/admin/profiling/stats` - Current profiling statistics
- `/admin/profiling/bottlenecks` - Performance bottlenecks
- `/admin/profiling/export` - Export profiling data

## HTTPS Enforcement
- Middleware in `universal_decoder_node/utils/https_middleware.py`
- Automatic HTTP->HTTPS redirects, path exclusions, security headers (HSTS, X-Content-Type-Options, etc.)
- Controlled via `ENFORCE_HTTPS=true`

## OpenTelemetry Tracing
Enable distributed tracing for end-to-end latency analysis:
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

resource = Resource.create({"service.name": "decoder-node"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)
```

## High-performance Serving (FastAPI/uvicorn)
Decoder nodes use FastAPI served via uvicorn with multiple workers:
```python
# Production startup
uvicorn cloud_decoder.optimized_decoder:app --host 0.0.0.0 --port 8001 --workers 4
```
- Tune worker count to GPU memory
- Combine with `--prefetch-vocab-groups` for warm start

## Adapter Downloads and Caching
```python
from huggingface_hub import snapshot_download
local_dir = os.getenv("ADAPTER_DIR", "/models/adapters")
repo_id = "org/uts-adapter-en-es"
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
```

## Best Practices
1. Enable memory monitoring in production
2. Configure appropriate thresholds for your hardware
3. Use profiling during development, disable in production
4. Analyze bottlenecks regularly
5. Export profiling data for trend tracking
6. Enforce HTTPS in production
7. Keep file fallback hot with periodic mirroring (`COORDINATOR_MIRROR_INTERVAL`)

## Troubleshooting

### Memory Issues
- High usage: reduce batch sizes or increase cleanup frequency
- OOM errors: check thresholds and cleanup frequency
- GPU leaks: check for tensor leaks

### Profiling Issues
- High overhead: profile only specific functions
- Missing data: check profiling is enabled
- Export errors: check output directory permissions
