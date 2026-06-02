# Redis Integration for Universal Translation System

This document explains how Redis is integrated into the Universal Translation System for improved scalability and reliability.

## Table of Contents
1. [Overview](#overview)
2. [Benefits](#benefits)
3. [Configuration](#configuration)
4. [Components Using Redis](#components-using-redis)
5. [Fallback Mechanisms](#fallback-mechanisms)
6. [Troubleshooting](#troubleshooting)

## Overview

Redis serves as a centralized storage solution for:
- Decoder pool management (via `utils.redis_manager.RedisManager`)
- Rate limiting across coordinator instances
- Token revocation (jti blacklist)
- Caching frequently used data

## Benefits

- **Distributed Decoder Pool**: Decoders across regions register centrally
- **Consistent Rate Limiting**: Enforced across all coordinator instances
- **Improved Reliability**: Automatic fallback to file-based storage + periodic Redis-to-disk mirroring
- **Horizontal Scaling**: Multiple coordinator instances share the same pool
- **Real-time Updates**: Changes immediately visible to all components

## Configuration

### Environment Variables
- `REDIS_URL`: Redis connection URL (e.g., `redis://localhost:6379/0`)
- `REDIS_PORT`: Redis port (default: `6379`)
- `REDIS_PASSWORD`: Redis password
- `COORDINATOR_MIRROR_INTERVAL`: Seconds between periodic Redis-to-disk mirrors (default: 60, minimum: 5)

### Setup Script
```bash
bash scripts/setup_redis.sh    # Install/start/stop/status with Docker fallback
```

### Docker Compose
The `docker-compose.yml` includes a Redis service:
```yaml
redis:
  image: redis:7-alpine
  container_name: translation_redis
  ports:
    - "${REDIS_PORT:-6379}:6379"
  volumes:
    - redis-data:/data
  command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Kubernetes
For K8s, Redis is configured via the Helm chart at `charts/uts/` or the `kubernetes/` manifests. The Helm chart includes Redis with PVC-backed persistence.

## Components Using Redis

### Coordinator
- Store/retrieve decoder pool configuration
- Track decoder health status
- Periodic disk mirroring via `DecoderPool.mirror_redis_to_disk()`

### Rate Limiter
- Track API usage across multiple coordinator instances
- Falls back to in-memory storage if Redis unavailable

### RedisManager
- Singleton in `utils/redis_manager.py` with double-checked locking
- Health checks with `redis_manager_healthy` Prometheus metric

## Fallback Mechanisms

1. **Coordinator**: Falls back to file-based storage (`config/decoder_pool.json`)
2. **Rate Limiter**: Falls back to in-memory storage
3. **Disk Mirroring**: `COORDINATOR_MIRROR_INTERVAL` controls periodic mirroring

## Troubleshooting

### Redis Connection Issues
```bash
redis-cli ping
echo $REDIS_URL
telnet <redis-host> 6379
docker logs translation_redis
```

### Data Consistency Issues
```bash
# Check Redis data
redis-cli get decoder_pool:nodes
# Compare with file
cat config/decoder_pool.json
```
