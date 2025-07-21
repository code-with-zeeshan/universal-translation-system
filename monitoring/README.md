# Monitoring Guide for Universal Translation System

This folder provides tools and instructions to monitor the health, performance, and usage of the entire Universal Translation System—including encoder, decoder, SDKs, and infrastructure.

---

## Overview
- Uses [Prometheus](https://prometheus.io/) for metrics collection
- Exposes metrics for translation requests, latency, GPU utilization, and more
- Can be extended to monitor all microservices, SDK endpoints, and infrastructure
- Integrates with Grafana for dashboards and alerting

---

## What Can Be Monitored?
- **Decoder (Cloud):**
  - Translation request count, latency, error rates
  - GPU utilization, memory usage
  - Active connections, health status
- **Encoder (Edge/SDK):**
  - (Optional) Local request count, latency (if running on server/desktop)
  - App-level metrics via SDK hooks
- **Infrastructure:**
  - Pod/container health (Kubernetes)
  - Node resource usage (CPU, RAM, GPU)
  - Network traffic

---

## How to Use

### 1. Integrate Metrics in Services
- Import and use `metrics_collector.py` in your FastAPI/Litserve decoder, or any Python microservice.
- Expose `/metrics` endpoint (already done in decoder Docker/K8s setup).
- For SDKs, add hooks to log or export metrics to a central endpoint if needed.

### 2. Prometheus Setup
- Use the provided `prometheus.yml` (see `cloud_decoder/` or create your own) to scrape metrics endpoints.
- Example scrape config:
  ```yaml
  scrape_configs:
    - job_name: 'decoder'
      static_configs:
        - targets: ['decoder-service:8000']
    - job_name: 'custom-metrics'
      static_configs:
        - targets: ['monitoring-service:9000']
  ```

### 3. Grafana Dashboards
- Connect Grafana to Prometheus to visualize:
  - Request rates, error rates
  - Latency histograms
  - GPU/CPU/memory utilization
  - Custom business metrics

### 4. Alerts
- Set up Prometheus alert rules for:
  - High error rates
  - High latency
  - GPU/CPU over-utilization
  - Service downtime

---

## Extending Monitoring
- Add new metrics in `metrics_collector.py` (e.g., for cache hits, queue lengths, etc.)
- Add exporters for node-level metrics (node_exporter, kube-state-metrics)
- For SDKs, send anonymized usage stats to a central endpoint (opt-in)
- Integrate with cloud provider monitoring (CloudWatch, Stackdriver, Azure Monitor)

---

## Example: Adding a Custom Metric
```python
from prometheus_client import Counter
custom_counter = Counter('custom_event_total', 'Description of custom event')
custom_counter.inc()
```

---

## File Reference
- `metrics_collector.py`: Core Prometheus metrics for translation system
- `prometheus.yml`: Example Prometheus scrape config (see cloud_decoder/)
- `README.md`: This guide

---

## See Also
- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)
- [DEPLOYMENT.md](../docs/DEPLOYMENT.md) 