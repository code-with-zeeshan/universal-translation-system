# Decoder Pool Management

This document explains how to add, remove, and manage decoders in the Universal Translation System's decoder pool.

- Note on ports: Coordinator default port in Docker Compose is 8002. Older docs or scripts may reference 5100 (deprecated). Use 8002 unless you explicitly run the Coordinator on 5100.

## Table of Contents
1. Overview
2. Decoder Pool Configuration
3. Storage Options
4. Adding a New Decoder
5. Removing a Decoder
6. Monitoring Decoder Health
7. Load Balancing
8. A/B Testing
9. Troubleshooting

## Overview
The system uses a pool of decoders managed by the Coordinator. Benefits:
- **Horizontal Scaling**: Add more decoders to handle increased load
- **High Availability**: Multiple decoders provide redundancy
- **Resource Optimization**: Distribute translation workloads across machines/regions/GPUs
- **Specialization**: Configure decoders for specific language pairs or domains
- **A/B Testing**: Compare model versions/configs safely

## Decoder Pool Configuration
The decoder pool can be configured dynamically via Coordinator or backed by Redis/file.

### Example (file-based) configuration (configs/decoder_pool.json)
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

### Configuration fields
- **node_id**: Unique identifier
- **endpoint**: HTTP endpoint for the decoder service
- **health_url**: Health endpoint for the decoder
- **status**: active | inactive | maintenance
- **languages**: optional; if omitted, treated as all supported
- **priority**: optional; higher values get fewer requests
- **region**: region label for routing
- **gpu_type**: informational (e.g., T4/A100)
- **capacity**: intended RPS capacity (coordinator may use for routing hints)
- **tags**: free-form labels
- **hot_adapters**: optional; currently loaded adapters

## Storage Options

### 1) Redis (recommended for production)
Set `REDIS_URL`, e.g.:
```bash
export REDIS_URL=redis://localhost:6379/0
```
In Docker Compose, this is already set for Coordinator as `redis://redis:6379/0`.

### 2) File-based (simple/local)
Default file path: `configs/decoder_pool.json`.
Customize via `POOL_CONFIG_PATH` env var.

### 3) etcd (experimental)
Advanced service discovery. Requires extra setup.

## Adding a New Decoder

### Method 1: Registration tool (recommended)
```bash
# Basic
python tools/register_decoder_node.py --endpoint http://your-decoder:8001

# Full options
python tools/register_decoder_node.py \
  --endpoint http://your-decoder:8001 \
  --region us-east-1 \
  --gpu_type T4 \
  --capacity 100 \
  --redis-url redis://localhost:6379/0 \
  --coordinator-url http://localhost:8002 \
  --api-key <your-admin-token-or-jwt> \
  --tags production,high-memory
```
The tool will:
1. Check decoder health
2. Generate `node_id`
3. Try coordinator API registration
4. Fallback to Redis, then file-based if needed

### Method 2: Update configuration file
1) Deploy decoder container:
```bash
docker build -t universal-decoder:latest -f docker/decoder.Dockerfile .
docker run -d --name decoder-3 --gpus all -p 8003:8001 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/vocabs:/app/vocabs" \
  -e DECODER_JWT_SECRET=replace-with-strong-secret \
  universal-decoder:latest
```
2) Add node to `configs/decoder_pool.json` and restart Coordinator:
```json
{
  "nodes": [
    { "node_id": "decoder-3", "endpoint": "http://decoder-3:8001", "health_url": "http://decoder-3:8001/health", "status": "active", "region": "us-west-1", "gpu_type": "T4", "capacity": 100, "tags": ["development"] }
  ]
}
```
```bash
docker compose restart coordinator
```

### Method 3: Coordinator API (v1 endpoints)
```bash
# Register decoder
curl -X POST http://localhost:8002/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "decoder-3",
    "endpoint": "http://decoder-3:8001",
    "region": "us-west-1",
    "gpu_type": "T4",
    "capacity": 100,
    "healthy": true,
    "load": 0
  }'

# Get info
curl http://localhost:8002/api/v1/node/decoder-3

# Update status
curl -X PUT http://localhost:8002/api/v1/node/decoder-3/status \
  -F healthy=true -F load=5

# Unregister
decode_id=decoder-3
curl -X DELETE http://localhost:8002/api/v1/unregister/$decode_id
```

## Removing or Disabling a Decoder
- File-based: remove entry from `configs/decoder_pool.json` and restart Coordinator
- API-based:
```bash
# Remove
debug_id=decoder-3
curl -X DELETE http://localhost:8002/api/decoders/$debug_id \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Set inactive
curl -X PATCH http://localhost:8002/api/decoders/decoder-3 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "inactive"}'
```

## Monitoring Decoder Health
- Coordinator periodically checks `/health` on each decoder (interval and retries are configurable)
- Unhealthy decoders are marked inactive; automatic reactivation when healthy
- Exposed metrics (Prometheus):
  - `coordinator_decoder_active`
  - `coordinator_decoder_load`
  - `coordinator_requests_total`
  - `coordinator_requests_errors`

## Load Balancing
- Least-loaded selection across healthy decoders
- Language support filtering
- Preference for decoders with required adapters already in memory
- Priority bias (higher priority → fewer requests)
- Circuit breaking to avoid unhealthy nodes
- Optional regional routing

## A/B Testing
Define experiments in the pool config under `ab_tests`:
```json
{
  "ab_tests": [
    {
      "id": "new-model-test",
      "description": "Testing new model version",
      "control_group": { "node_ids": ["decoder-1"] },
      "test_group": { "node_ids": ["decoder-2"] },
      "traffic_split": 0.5,
      "active": true
    }
  ]
}
```
Monitor metrics to compare groups.

## Troubleshooting

### Decoder not receiving requests
```bash
# Health
curl http://localhost:8002/api/decoders/decoder-3/health -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
# Registration
curl http://localhost:8002/api/decoders -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
# Logs
docker logs decoder-3
# From coordinator container, check network
docker compose exec coordinator wget -q --spider http://decoder-3:8001/health || echo "Connection failed"
```

### Coordinator cannot connect to decoder
- Verify endpoint URL is correct and reachable from Coordinator container
- Check firewall/security groups
- Ensure decoder is running and healthy

### Performance issues
- Check GPU utilization (`nvidia-smi`)
- Inspect decoder `/metrics`
- Tune batch size/workers
- Verify hot adapters for frequently used languages

---
Deprecated reference: Coordinator port `5100` in older docs. Use `8002` by default in this repository setup.