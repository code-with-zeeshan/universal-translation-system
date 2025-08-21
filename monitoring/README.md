# Monitoring Guide for Universal Translation System

This folder provides tools and instructions to monitor the health, performance, and usage of the entire Universal Translation System—including encoder, decoder, SDKs, and infrastructure.

---

## Overview
- Uses [Prometheus](https://prometheus.io/) for metrics collection
- Exposes metrics for translation requests, latency, GPU utilization, and more
- Can be extended to monitor all microservices, SDK endpoints, and infrastructure
- Integrates with Grafana for dashboards and alerting
- Implements circuit breaker pattern for resilience
- Provides detailed metrics for all components

---

## What Can Be Monitored?

### Decoder (Cloud)
- Translation request count, latency, error rates
- GPU utilization, memory usage
- Active connections, health status
- Queue size and processing time
- Circuit breaker state and events

### Encoder (Edge/SDK)
- Local request count, latency (if running on server/desktop)
- App-level metrics via SDK hooks
- Encoding size and compression ratio
- Vocabulary cache hits/misses
- Memory usage and performance

### Coordinator
- Active decoders and routing decisions
- Load balancing effectiveness
- Circuit breaker states for all decoders
- Request routing patterns
- Error rates and recovery attempts

### Vocabulary System
- Cache size and hit/miss rates
- Download success/failure rates
- Memory usage per language
- Tokenization performance

### Infrastructure
- Pod/container health (Kubernetes)
- Node resource usage (CPU, RAM, GPU)
- Network traffic and latency
- Disk usage and I/O performance

---

## How to Use

### 1. Integrate Enhanced Metrics in Services
- Import and use `enhanced_metrics.py` in your FastAPI/Litserve decoder, or any Python microservice.
- Expose `/metrics` endpoint (already done in decoder Docker/K8s setup).
- For SDKs, add hooks to log or export metrics to a central endpoint if needed.
- Use the provided decorators for automatic latency tracking:

```python
from monitoring.enhanced_metrics import track_async_latency

@track_async_latency(source_lang="en", target_lang="es", sdk_type="python", component="encoder")
async def encode_text(text):
    # Your encoding logic here
    return encoded_data
```

### 2. Prometheus Setup
- Use the provided `prometheus.yml` (see `cloud_decoder/` or create your own) to scrape metrics endpoints.
- Example scrape config:
  ```yaml
  scrape_configs:
    - job_name: 'decoder'
      static_configs:
        - targets: ['decoder-service:8001']
    - job_name: 'coordinator'
      static_configs:
        - targets: ['coordinator-service:8002']
    - job_name: 'encoder'
      static_configs:
        - targets: ['encoder-service:8000']
    - job_name: 'system-metrics'
      static_configs:
        - targets: ['monitoring-service:9000']
  ```

### 3. Grafana Dashboards
- Connect Grafana to Prometheus to visualize metrics
- Use the provided comprehensive dashboard:
  - Import `grafana/dashboards/comprehensive.json` into your Grafana instance
  - The dashboard includes panels for:
    - Translation requests and latency
    - System resource usage (CPU, memory, GPU)
    - Component-specific metrics (encoder, decoder, vocabulary)
    - Error monitoring and circuit breaker status
    - Coordinator routing decisions

### 4. System Metrics Collection
- Start the system metrics collection service:

```python
from monitoring.enhanced_metrics import start_system_metrics_collection

# For synchronous applications
metrics_thread = start_system_metrics_collection(interval=15)  # 15 seconds interval

# For asynchronous applications
async def start_metrics():
    metrics_task = await start_async_system_metrics_collection(interval=15)
    return metrics_task
```

### 5. Circuit Breaker Integration
- Integrate the circuit breaker pattern with your coordinator:

```python
from coordinator.circuit_breaker import CircuitBreaker

# Create a circuit breaker for each decoder
circuit_breaker = CircuitBreaker(
    name="decoder-1",
    failure_threshold=5,
    recovery_timeout=30,
    timeout=10
)

# Use the circuit breaker to execute functions
async def call_decoder(decoder_url, data):
    async def _call():
        # Your decoder call logic here
        return await http_client.post(decoder_url, json=data)
    
    try:
        return await circuit_breaker.execute(_call)
    except Exception as e:
        # Handle the error or fall back to another decoder
        logger.error(f"Decoder call failed: {e}")
        raise
```

### 6. Alerts
- Set up Prometheus alert rules for:
  - High error rates: `error_count_total[5m] > 10`
  - High latency: `histogram_quantile(0.95, sum(rate(translation_latency_seconds_bucket[5m])) by (le)) > 1`
  - GPU over-utilization: `decoder_gpu_utilization_percent > 90`
  - Circuit breaker open: `circuit_breaker_state > 1`
  - Service downtime: `up == 0`
  - Memory leaks: `rate(system_memory_usage_bytes[30m]) > 0`

---

## Extending Monitoring
- Add new metrics in `enhanced_metrics.py` (e.g., for cache hits, queue lengths, etc.)
- Add exporters for node-level metrics (node_exporter, kube-state-metrics)
- For SDKs, send anonymized usage stats to a central endpoint (opt-in)
- Integrate with cloud provider monitoring (CloudWatch, Stackdriver, Azure Monitor)
- Create custom dashboards for specific use cases or components

---

## Example: Adding a Custom Metric
```python
from prometheus_client import Counter, Gauge, Histogram

# Counter for events that increment
custom_counter = Counter('custom_event_total', 'Description of custom event', ['label1', 'label2'])
custom_counter.labels(label1='value1', label2='value2').inc()

# Gauge for values that can go up and down
custom_gauge = Gauge('custom_value', 'Description of custom value', ['label1'])
custom_gauge.labels(label1='value1').set(42)

# Histogram for measuring distributions
custom_histogram = Histogram('custom_duration_seconds', 'Description of duration', ['label1'], 
                            buckets=(0.1, 0.5, 1, 5, 10, 30, 60))
custom_histogram.labels(label1='value1').observe(0.7)  # Records in the 0.5-1 bucket
```

## Environment Variable Configuration

The monitoring system can be configured using environment variables:

```bash
# Prometheus endpoint configuration
PROMETHEUS_PORT=9090
METRICS_PATH=/metrics

# Logging level
MONITORING_LOG_LEVEL=INFO

# Metrics collection interval (seconds)
METRICS_COLLECTION_INTERVAL=15

# Enable/disable specific metrics
ENABLE_SYSTEM_METRICS=true
ENABLE_GPU_METRICS=true
ENABLE_VOCABULARY_METRICS=true
```

See [docs/environment-variables.md](../docs/environment-variables.md) for a complete list of environment variables.

---

## Monitoring Best Practices

### 1. Focus on the Four Golden Signals
- **Latency**: How long it takes to serve a request
- **Traffic**: How much demand is being placed on your system
- **Errors**: The rate of failed requests
- **Saturation**: How "full" your service is (resource utilization)

### 2. Use Labels Effectively
- Add labels for source/target languages, component names, and error types
- Avoid high cardinality labels (e.g., user IDs, session IDs)
- Use consistent label naming across metrics

### 3. Set Up Proper Alerting
- Alert on symptoms, not causes
- Set thresholds based on historical data
- Implement different severity levels
- Ensure alerts are actionable

### 4. Monitor from the User's Perspective
- Track end-to-end latency, not just component latency
- Monitor error rates as experienced by users
- Set up synthetic monitoring for key workflows

---

## File Reference
- `enhanced_metrics.py`: Comprehensive Prometheus metrics for all components
- `prometheus.yml`: Example Prometheus scrape config (see cloud_decoder/)
- `grafana/dashboards/comprehensive.json`: Complete Grafana dashboard
- `README.md`: This guide

---

## See Also
- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)
- [DEPLOYMENT.md](../docs/DEPLOYMENT.md)
- [TROUBLESHOOT.md](../docs/TROUBLESHOOT.md)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)