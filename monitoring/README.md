# Monitoring Guide for Universal Translation System

This folder provides tools and instructions to monitor the health, performance, and usage of the entire Universal Translation System.

---

## Overview
- Uses Prometheus for metrics collection
- Exposes metrics for translation requests, latency, GPU utilization, and more
- Integrates with Grafana for dashboards and alerting
- Alerting rules in `prometheus/rules/alerting_rules.yml`
- Recording rules in `prometheus/rules/recording_rules.yml`

---

## What Can Be Monitored?

### Decoder (Cloud)
- Translation request count, latency, error rates
- GPU utilization, memory usage
- Active connections, health status
- Circuit breaker state

### Encoder (Edge/SDK)
- Local request count, latency
- Encoding size and compression ratio
- Vocabulary cache hits/misses

### Coordinator
- Active decoders and routing decisions
- Load balancing effectiveness
- Circuit breaker states
- Error rates and recovery attempts

### Vocabulary System
- Cache size and hit/miss rates
- Download success/failure rates
- Tokenization performance

---

## How to Use

### 1. Prometheus Setup
The canonical config is at `monitoring/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'decoder'
    static_configs:
      - targets: ['decoder-service:8001']
  - job_name: 'coordinator'
    static_configs:
      - targets: ['coordinator-service:5100']
  - job_name: 'encoder'
    static_configs:
      - targets: ['encoder-service:8000']
```

### 2. Grafana Dashboards
- Connect Grafana to Prometheus
- Import `grafana/dashboards/comprehensive.json`
- Panels for translation requests, latency, system resources, component metrics

### 3. Alerts
Alerting rules in `prometheus/rules/alerting_rules.yml`:
- ServiceDown: if target is unreachable for >1m
- HighErrorRate: >10% errors over 5m
- HighLatency: p95 latency >1s
- HighCPUUsage: >80% over 10m
- HighMemoryUsage: >90% over 10m

### 4. Environment Variable Configuration
```bash
PROMETHEUS_PORT=9090
METRICS_PATH=/metrics
METRICS_COLLECTION_INTERVAL=15
ENABLE_SYSTEM_METRICS=true
ENABLE_GPU_METRICS=true
ENABLE_VOCABULARY_METRICS=true
```

---

## File Reference
- `prometheus/prometheus.yml` -- Scrape configuration
- `prometheus/rules/alerting_rules.yml` -- Alert rules
- `prometheus/rules/recording_rules.yml` -- Recording rules
- `grafana/dashboards/comprehensive.json` -- Grafana dashboard
- `metrics.py` -- All Prometheus metrics (enhanced metrics merged)
- `metrics_collector.py` -- Metrics collection helpers

---

## See Also
- [Prometheus Docs](https://prometheus.io/docs/)
- [Grafana Docs](https://grafana.com/docs/)
- [DEPLOYMENT.md](../docs/DEPLOYMENT.md)
- [TROUBLESHOOT.md](../docs/TROUBLESHOOT.md)
