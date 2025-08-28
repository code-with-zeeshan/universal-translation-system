# Performance Optimization

This document describes the performance optimization features of the Universal Translation System, including memory management, profiling, and bottleneck detection.

## Memory Management

The Universal Translation System includes a comprehensive memory management system to ensure optimal resource utilization and prevent out-of-memory errors.

### Key Features

- **Real-time Memory Monitoring:** Continuous monitoring of system and GPU memory usage
- **Automatic Cleanup:** Configurable thresholds for triggering memory cleanup
- **Model Optimization:** Automatic optimization of models for inference
- **Alert System:** Configurable alerts for memory usage thresholds
- **Resource Recommendations:** Intelligent recommendations based on usage patterns

### Configuration

Memory management can be configured through environment variables or configuration files:

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

The memory manager is automatically initialized in the decoder service and can be accessed through the `memory_manager` singleton:

```python
# If using the packaged decoder node
from universal_decoder_node.utils.memory_manager import MemoryManager
# If importing directly from repo source, adjust PYTHONPATH accordingly
# from universal-decoder-node.universal_decoder_node.utils.memory_manager import MemoryManager

# Get memory manager instance
memory_manager = MemoryManager.get_instance()

# Get current memory stats
memory_stats = memory_manager.get_memory_stats()

# Perform manual cleanup if needed
memory_manager.cleanup()
```

### Memory Monitoring Endpoint

The decoder service includes a `/status` endpoint that provides memory usage information:

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

Note: The memory management, profiling utilities, and HTTPS middleware referenced below live in the universal-decoder-node package under universal_decoder_node/utils. If you are developing inside this monorepo without installing the package, import paths should use universal-decoder-node/universal_decoder_node/utils/... or update PYTHONPATH accordingly.

The profiling system provides insights into performance bottlenecks and helps optimize the system for maximum throughput.

### Key Features

- **Function Profiling:** Detailed timing of function execution
- **Section Profiling:** Granular profiling of code sections
- **Bottleneck Detection:** Automatic identification of performance bottlenecks
- **Export Capabilities:** Multiple export formats for analysis (JSON, CSV, TXT)
- **History Tracking:** Tracking of performance trends over time

### Configuration

Profiling can be configured through environment variables or configuration files:

```yaml
profiling:
  enable_profiling: true
  profile_output_dir: "profiles"
  bottleneck_threshold_ms: 100.0
  export_format: "json"
```

### Usage

The profiling system provides decorators and context managers for profiling code:

```python
# If using the packaged decoder node
from universal_decoder_node.utils.profiler import profile, profile_section, function_profiler
# If importing directly from repo source, adjust PYTHONPATH accordingly
# from universal-decoder-node.universal_decoder_node.utils.profiler import profile, profile_section, function_profiler

# Profile a function
@profile
def my_function():
    # Function code here
    pass

# Profile a section of code
def process_data(data):
    # Some code here
    
    with profile_section("data_processing"):
        # Code to profile
        result = process_complex_data(data)
    
    # More code here
    return result

# Get profiling stats
stats = function_profiler.get_stats()

# Identify bottlenecks
bottlenecks = function_profiler.identify_bottlenecks()

# Export profiling data
function_profiler.export_stats(filepath="profiles/my_profile.json")
```

### Profiling Endpoints

The decoder service includes endpoints for accessing profiling data:

- `/admin/profiling/stats` - Get current profiling statistics
- `/admin/profiling/bottlenecks` - Identify performance bottlenecks
- `/admin/profiling/export` - Export profiling data

## HTTPS Enforcement

The system includes configurable HTTPS enforcement to ensure secure communication.

- The middleware and helpers are implemented in universal_decoder_node/utils/https_middleware.py.
- It provides automatic HTTP→HTTPS redirects, path exclusions (e.g., /health, /metrics), and security headers (HSTS, X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy).
- You can control enforcement via ENFORCE_HTTPS=true and a custom port via the https_port parameter.

### Key Features

- **Configurable Enforcement:** Enable or disable HTTPS enforcement
- **Path Exclusions:** Exclude specific paths from HTTPS enforcement
- **Security Headers:** Comprehensive set of security headers
- **Configurable HTTPS Port:** Specify the HTTPS port

### Configuration

HTTPS enforcement can be configured through environment variables or configuration files:

```yaml
https:
  enforce: true
  port: 443
```

### Usage

HTTPS enforcement is automatically applied to the FastAPI application:

```python
# If using the packaged decoder node
from universal_decoder_node.utils.https_middleware import add_https_middleware
# If importing directly from repo source, adjust PYTHONPATH accordingly
# from universal-decoder-node.universal_decoder_node.utils.https_middleware import add_https_middleware

# Add HTTPS middleware to FastAPI app
add_https_middleware(app, enforce_https=True, https_port=443)
```

## OpenTelemetry Tracing

Enable distributed tracing in the decoder service to analyze end-to-end latency and pinpoint bottlenecks.

### Minimal setup

```python
# otel_setup.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

resource = Resource.create({"service.name": "decoder-node"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"))
)
trace.set_tracer_provider(provider)

# Instrument FastAPI/Litserve app
FastAPIInstrumentor.instrument_app(app)
```

- Configure OTEL_EXPORTER_OTLP_ENDPOINT or the explicit endpoint above.
- Use Grafana Tempo/Jaeger as tracing backends via OpenTelemetry Collector.

## High-performance Serving (Litserve)

Decoder nodes use Litserve (FastAPI-based) for high-throughput, low-latency serving with batching:

```python
from litserve import LitServer
from my_model import DecoderModel

model = DecoderModel()
server = LitServer(model, max_batch_size=8, timeout=10)
app = server.to_fastapi()  # Your FastAPI app object
```

- Tune max_batch_size and timeout to balance latency/throughput.
- Combine with uvicorn workers and GPU-per-process pinning in production.

## Adapter Downloads and Caching (Hugging Face Hub)

Decoder nodes can fetch per-language adapters or models at startup or on-demand:

```python
import os
from huggingface_hub import snapshot_download

local_dir = os.getenv("ADAPTER_DIR", "/models/adapters")
repo_id = "org/uts-adapter-en-es"
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
```

- Control cache via HF_HOME or local_dir.
- Pre-warm adapters on startup to avoid cold latency spikes.

## Request Batching and Timeouts

- Use Litserve batching knobs (max_batch_size, timeout) to aggregate small requests when load permits.
- Ensure client-side timeouts accommodate batching delay under load.

## Getting Started: Wiring Memory, Profiling, HTTPS

Below is a minimal example showing how to integrate memory manager, profiling, and HTTPS middleware into a FastAPI/Litserve app:

```python
# app.py
from fastapi import FastAPI
from universal_decoder_node.utils.memory_manager import MemoryManager
from universal_decoder_node.utils.profiler import profile, profile_section, function_profiler
from universal_decoder_node.utils.https_middleware import add_https_middleware

app = FastAPI(title="Decoder Node")

# 1) Add HTTPS middleware (configurable via env ENFORCE_HTTPS)
add_https_middleware(app, enforce_https=True, https_port=443)

# 2) Initialize memory manager (singleton)
memory_manager = MemoryManager.get_instance()

# 3) Example endpoint with profiling
@app.get("/decode")
@profile
async def decode():
    with profile_section("prepare_inputs"):
        # ... prepare inputs
        pass
    with profile_section("inference"):
        # ... run model
        pass
    with profile_section("postprocess"):
        # ... post-process outputs
        pass
    return {"ok": True}

# 4) Health and metrics endpoints are typically provided elsewhere in the service
#    but can be added here as needed:
@app.get("/health")
async def health():
    return {"status": "healthy", "memory": memory_manager.get_memory_stats()}
```

- For OpenTelemetry tracing, see the section above and initialize instrumentation after app is created.

## Best Practices

1. **Enable Memory Monitoring:** Always enable memory monitoring in production environments to prevent out-of-memory errors.

2. **Configure Appropriate Thresholds:** Set memory thresholds based on your specific hardware and workload.

3. **Use Profiling Selectively:** Enable profiling during development and testing, but consider disabling it in production for maximum performance.

4. **Analyze Bottlenecks Regularly:** Regularly review profiling data to identify and address performance bottlenecks.

5. **Export Profiling Data:** Export profiling data for offline analysis and trend tracking.

6. **Enforce HTTPS:** Always enforce HTTPS in production environments to ensure secure communication.

7. **Coordinate State Efficiently:** When using Redis-backed coordinator, keep file fallback hot with periodic mirroring (controlled by `COORDINATOR_MIRROR_INTERVAL`, min 5s) to avoid cold starts and improve reload times.

## Troubleshooting

### Memory Issues

- **High Memory Usage:** If memory usage is consistently high, consider reducing batch sizes or enabling more aggressive cleanup.
- **Out-of-Memory Errors:** If you encounter OOM errors, check the memory thresholds and consider increasing cleanup frequency.
- **GPU Memory Leaks:** If GPU memory usage increases over time, check for tensor leaks and ensure proper cleanup.

### Profiling Issues

- **High Overhead:** If profiling adds significant overhead, consider profiling only specific functions or sections.
- **Missing Data:** If profiling data is incomplete, check that profiling is enabled and that the code is properly instrumented.
- **Export Errors:** If export fails, check that the output directory exists and is writable.

### HTTPS Issues

- **Redirect Loops:** If you encounter redirect loops, check that your load balancer or proxy is properly configured.
- **Mixed Content:** If you see mixed content warnings, ensure all resources are served over HTTPS.
- **Certificate Errors:** If you see certificate errors, check that your SSL certificate is valid and properly installed.