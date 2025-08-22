# Decoder Pool Management

This document explains how to add, remove, and manage decoders in the Universal Translation System's decoder pool.

## Table of Contents

1. [Overview](#overview)
2. [Decoder Pool Configuration](#decoder-pool-configuration)
3. [Adding a New Decoder](#adding-a-new-decoder)
4. [Removing a Decoder](#removing-a-decoder)
5. [Monitoring Decoder Health](#monitoring-decoder-health)
6. [Load Balancing](#load-balancing)
7. [Troubleshooting](#troubleshooting)

## Overview

The Universal Translation System uses a pool of decoders managed by the coordinator service. This architecture allows for:

- **Horizontal Scaling**: Add more decoders to handle increased load
- **High Availability**: Multiple decoders provide redundancy
- **Resource Optimization**: Distribute translation workloads across multiple machines
- **Specialized Decoders**: Configure decoders for specific language pairs or domains

## Decoder Pool Configuration

The decoder pool is configured in the coordinator service using a JSON configuration file. By default, this file is located at `config/decoder_pool.json`.

### Example Configuration

```json
{
  "decoders": [
    {
      "id": "decoder-1",
      "url": "http://decoder-service-1:8001",
      "health_url": "http://decoder-service-1:8001/health",
      "status": "active",
      "languages": ["en", "es", "fr", "de"],
      "priority": 1
    },
    {
      "id": "decoder-2",
      "url": "http://decoder-service-2:8001",
      "health_url": "http://decoder-service-2:8001/health",
      "status": "active",
      "languages": ["en", "zh", "ja", "ko"],
      "priority": 1
    }
  ]
}
```

### Configuration Fields

- **id**: Unique identifier for the decoder
- **url**: HTTP endpoint for the decoder service
- **health_url**: Health check endpoint for the decoder
- **status**: Current status (active, inactive, maintenance)
- **languages** (optional): List of supported languages (if not specified, all languages are supported)
- **priority** (optional): Priority level (higher numbers get fewer requests)

## Adding a New Decoder

### Method 1: Update Configuration File

1. Deploy a new decoder service using the provided Dockerfile:

   ```bash
   # Build the decoder image
   docker build -t universal-decoder:latest -f docker/decoder.Dockerfile .

   # Run the decoder container
   docker run -d --name decoder-3 -p 8003:8001 \
     -v /path/to/models:/app/models \
     -v /path/to/vocabs:/app/vocabs \
     -e DECODER_JWT_SECRET=your-jwt-secret \
     universal-decoder:latest
   ```

2. Add the new decoder to the `decoder_pool.json` file:

   ```json
   {
     "decoders": [
       // ... existing decoders ...
       {
         "id": "decoder-3",
         "url": "http://decoder-3:8001",
         "health_url": "http://decoder-3:8001/health",
         "status": "active"
       }
     ]
   }
   ```

3. Restart the coordinator service to apply the changes:

   ```bash
   docker restart coordinator
   ```

### Method 2: Use the Coordinator API

The coordinator provides a REST API for managing the decoder pool dynamically:

1. Add a new decoder:

   ```bash
   curl -X POST http://coordinator:5100/api/decoders \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "decoder-3",
       "url": "http://decoder-3:8001",
       "health_url": "http://decoder-3:8001/health",
       "status": "active"
     }'
   ```

2. Verify the decoder was added:

   ```bash
   curl -X GET http://coordinator:5100/api/decoders \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```

## Removing a Decoder

### Method 1: Update Configuration File

1. Edit the `decoder_pool.json` file and remove the decoder entry
2. Restart the coordinator service

### Method 2: Use the Coordinator API

1. Remove a decoder:

   ```bash
   curl -X DELETE http://coordinator:5100/api/decoders/decoder-3 \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```

2. Alternatively, set a decoder to inactive without removing it:

   ```bash
   curl -X PATCH http://coordinator:5100/api/decoders/decoder-3 \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "status": "inactive"
     }'
   ```

## Monitoring Decoder Health

The coordinator automatically monitors the health of all decoders in the pool:

1. Health checks are performed periodically (default: every 30 seconds)
2. If a decoder fails health checks, it's marked as unhealthy
3. After multiple failures (default: 3), the decoder is marked as inactive
4. The coordinator will automatically reactivate the decoder when it becomes healthy again

### Health Check Endpoint

Each decoder must implement a `/health` endpoint that returns:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_utilization": 0.45,
  "memory_utilization": 0.65,
  "uptime": 3600
}
```

## Load Balancing

The coordinator uses a load balancing algorithm to distribute requests across the decoder pool:

1. **Least Loaded**: Requests are sent to the decoder with the lowest current load
2. **Language Support**: Requests are only sent to decoders that support the requested language pair
3. **Priority**: Decoders with higher priority values receive fewer requests
4. **Circuit Breaking**: Unhealthy decoders are temporarily removed from the pool

## Troubleshooting

### Decoder Not Receiving Requests

1. Check the decoder's health status:

   ```bash
   curl -X GET http://coordinator:5100/api/decoders/decoder-3/health \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```

2. Verify the decoder is marked as active:

   ```bash
   curl -X GET http://coordinator:5100/api/decoders/decoder-3 \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```

3. Check the decoder's logs:

   ```bash
   docker logs decoder-3
   ```

### Coordinator Cannot Connect to Decoder

1. Verify network connectivity:

   ```bash
   # From coordinator container
   wget -q --spider http://decoder-3:8001/health || echo "Connection failed"
   ```

2. Check firewall settings and network policies
3. Verify the decoder service is running:

   ```bash
   docker ps | grep decoder-3
   ```

### Decoder Performance Issues

1. Check GPU utilization:

   ```bash
   nvidia-smi
   ```

2. Monitor decoder metrics:

   ```bash
   curl -X GET http://decoder-3:8001/metrics
   ```

3. Consider adjusting batch size and other performance parameters in the decoder configuration