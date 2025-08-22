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

The Universal Translation System now uses Redis as a centralized storage solution for:

- Decoder pool management
- Rate limiting
- Caching frequently used data

This integration enables better scalability across multiple regions and provides a more robust system for distributed deployments.

## Benefits

- **Distributed Decoder Pool**: Decoders running in any region can register with the central Redis instance
- **Consistent Rate Limiting**: Rate limits are enforced consistently across all coordinator instances
- **Improved Reliability**: Automatic fallback to file-based storage if Redis is unavailable
- **Horizontal Scaling**: Multiple coordinator instances can share the same decoder pool
- **Real-time Updates**: Changes to the decoder pool are immediately visible to all components

## Configuration

### Environment Variables

The following environment variables can be used to configure Redis:

- `REDIS_URL`: Redis connection URL (e.g., `redis://localhost:6379/0`)
- `REDIS_PORT`: Redis port (default: `6379`)
- `REDIS_PASSWORD`: Redis password (if authentication is enabled)

### Docker Compose

The `docker-compose.yml` file includes a Redis service:

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
  restart: unless-stopped
```

### Kubernetes

For Kubernetes deployments, use the following configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
        command: ["redis-server", "--appendonly", "yes", "--maxmemory", "1gb", "--maxmemory-policy", "allkeys-lru"]
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

## Components Using Redis

### Coordinator

The coordinator service uses Redis to:

1. Store and retrieve the decoder pool configuration
2. Track decoder health status
3. Store A/B test configurations

If Redis is unavailable, it falls back to using the local file system.

### Rate Limiter

The rate limiter uses Redis to:

1. Track API usage across multiple coordinator instances
2. Enforce consistent rate limits
3. Collect usage statistics

If Redis is unavailable, it falls back to in-memory storage (note: this will not be consistent across multiple instances).

### Register Decoder Node Tool

The `register_decoder_node.py` tool can register decoders with:

1. The coordinator API directly (preferred)
2. Redis (if coordinator API is unavailable)
3. Local file system (as a last resort)

## Fallback Mechanisms

The system implements graceful fallbacks at multiple levels:

1. **Coordinator Fallback**: If Redis is unavailable, the coordinator falls back to file-based storage
2. **Rate Limiter Fallback**: If Redis is unavailable, the rate limiter falls back to in-memory storage
3. **Registration Fallback**: The registration tool tries multiple methods in order of preference

This ensures that the system remains operational even if Redis is temporarily unavailable.

## Troubleshooting

### Redis Connection Issues

If you experience Redis connection issues:

1. Verify that Redis is running: `redis-cli ping`
2. Check the Redis URL: `echo $REDIS_URL`
3. Verify network connectivity: `telnet <redis-host> 6379`
4. Check Redis logs: `docker logs translation_redis`

### Data Consistency Issues

If you notice inconsistencies in the decoder pool:

1. Check Redis data: `redis-cli get decoder_pool:nodes`
2. Compare with file data: `cat configs/decoder_pool.json`
3. Restart the coordinator to force a reload: `docker restart coordinator`

### Performance Issues

If Redis is causing performance bottlenecks:

1. Monitor Redis performance: `redis-cli info`
2. Consider increasing Redis memory: Update `--maxmemory` in `docker-compose.yml`
3. For high-traffic deployments, consider Redis Cluster or Redis Sentinel