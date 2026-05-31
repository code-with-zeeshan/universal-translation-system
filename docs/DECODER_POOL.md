# Decoder Pool Management

This document explains how to add, remove, and manage decoders in the Universal Translation System's decoder pool.

- **Ports**: Coordinator default port in Docker Compose is 8002; under Helm/K8s it is 5100. Decoder default is 8001.
- **File-based pool**: `configs/decoder_pool.json` (overridable via `POOL_CONFIG_PATH`)

## Table of Contents
1. [Overview](#overview)
2. [Decoder Pool Configuration](#decoder-pool-configuration)
3. [Storage Options](#storage-options)
4. [Adding a New Decoder](#adding-a-new-decoder)
5. [Removing a Decoder](#removing-a-decoder)
6. [Monitoring Decoder Health](#monitoring-decoder-health)
7. [Load Balancing](#load-balancing)
8. [A/B Testing](#ab-testing)
9. [Troubleshooting](#troubleshooting)

## Overview
The system uses a pool of decoders managed by the Coordinator (see `coordinator/advanced_coordinator.py`):
- **Horizontal Scaling**: Add more decoders for increased load
- **High Availability**: Multiple decoders provide redundancy
- **Resource Optimization**: Distribute workloads across machines/regions/GPUs
- **A/B Testing**: Compare model versions safely

## Decoder Pool Configuration
The decoder pool can be configured dynamically via Coordinator or backed by Redis/file.

### Example configuration (`configs/decoder_pool.json`)
```json
{
  "nodes": [
    {
      "node_id": "decoder-1",
      "endpoint": "http://decoder:8001",
      "health_url": "http://decoder:8001/health",
      "status": "active",
      "languages": ["en", "es", "fr", "de"],
      "priority": 1,
      "region": "us-east-1",
      "gpu_type": "T4",
      "capacity": 100,
      "tags": ["production", "general-purpose"]
    }
  ],
  "ab_tests": []
}
```

## Storage Options

### 1) Redis (recommended for production)
Set `REDIS_URL`, e.g.: `redis://redis:6379/0` (Docker Compose auto-configures this).

### 2) File-based (simple/local)
Default file: `configs/decoder_pool.json`. Customize via `POOL_CONFIG_PATH` env var.
The coordinator periodically mirrors Redis state to this file (`COORDINATOR_MIRROR_INTERVAL`, default 60s).

### 3) etcd (experimental)
Advanced service discovery via `USE_ETCD=true`.

## Adding a New Decoder

### Method 1: Register via setup script
```bash
bash scripts/setup_serving.sh --register --endpoint http://your-decoder:8001 --coordinator http://coordinator:8002
```

### Method 2: Coordinator API (v1 endpoints)
```bash
curl -X POST http://localhost:8002/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{"node_id": "decoder-3", "endpoint": "http://decoder-3:8001", "region": "us-west-1", "gpu_type": "T4", "capacity": 100, "healthy": true, "load": 0}'
```

### Method 3: Update configuration file
1. Deploy decoder container:
```bash
docker run -d --name decoder-3 --gpus all -p 8003:8001 \
  -v "$PWD/models:/app/models" -v "$PWD/vocabs:/app/vocabs" \
  -e DECODER_JWT_SECRET=replace-with-strong-secret \
  universal-decoder:latest
```
2. Add node to `configs/decoder_pool.json` and restart Coordinator.

## Removing or Disabling a Decoder
```bash
# API-based removal
curl -X DELETE http://localhost:8002/api/decoders/decoder-3 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Set inactive
curl -X PATCH http://localhost:8002/api/decoders/decoder-3 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "inactive"}'
```

## Monitoring Decoder Health
- Coordinator periodically checks `/health` on each decoder
- Unhealthy decoders marked inactive; automatic reactivation when healthy
- Prometheus metrics:
  - `coordinator_decoder_active`
  - `coordinator_decoder_load`
  - `coordinator_requests_total`
  - `coordinator_requests_errors`

## Load Balancing
- Least-loaded selection across healthy decoders
- Language support filtering
- Preference for decoders with required adapters already in memory
- Priority bias (higher priority -> fewer requests)
- Circuit breaking to avoid unhealthy nodes
- Optional regional routing

## A/B Testing
Define experiments in pool config under `ab_tests`:
```json
{
  "ab_tests": [
    {
      "id": "new-model-test",
      "control_group": { "node_ids": ["decoder-1"] },
      "test_group": { "node_ids": ["decoder-2"] },
      "traffic_split": 0.5,
      "active": true
    }
  ]
}
```

## Troubleshooting

### Decoder not receiving requests
```bash
curl http://localhost:8002/api/decoders/decoder-3/health -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
docker logs decoder-3
docker compose exec coordinator wget -q --spider http://decoder-3:8001/health
```

### Coordinator cannot connect to decoder
- Verify endpoint URL reachable from Coordinator
- Check firewall/security groups
- Ensure decoder is running and healthy

### Performance issues
- Check GPU utilization (`nvidia-smi`)
- Inspect decoder `/metrics`
- Tune batch size/workers
- Verify hot adapters for frequently used languages
