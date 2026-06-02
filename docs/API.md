# API Documentation

## Overview
Two public surfaces exist:
- Decoder Node API (FastAPI/uvicorn serving translations)
- Coordinator API (routes requests to least-loaded healthy decoder nodes)

Use binary `/decode` for highest performance.

---

## Base URLs
- Decoder Node: http(s)://<decoder-host>:<port> (default 8001)
- Coordinator: http(s)://<coordinator-host>:<port> (default 5100)

---

## Decoder Node API
FastAPI app in `cloud_decoder/optimized_decoder.py`, served via uvicorn.

### POST /decode
- Purpose: Decode locally encoded embeddings (binary) to a translation.
- Headers:
  - `Content-Type: application/octet-stream`
  - `X-Target-Language: <code>` (required)
  - `X-Domain: <domain>` (optional, e.g., medical, legal)
  - `X-Client-ID: <id>` (optional; used for rate limiting)
- Body: Binary payload produced by the edge encoder, LZ4 + MsgPack compressed.
- Response: `application/json`
```json
{
  "translation": "Hola mundo",
  "target_language": "es"
}
```

Binary Payload Format (see `docs/schemas/binary_payload_schema.md`):
- LZ4-compressed MsgPack with fields: `tokens`, `embeddings`, `metadata`

### GET /health
- Returns basic health info.
```json
{ "status": "healthy", "device": "cuda", "adapters": ["es", "fr"], "vocab_packs": ["latin"] }
```

### GET /status
- Returns extended status and metrics summary.
```json
{
  "model_version": "1.0.0",
  "healthy": true,
  "metrics": { },
  "vocabulary": { }
}
```

### GET /metrics
- Prometheus metrics endpoint.
- Response: `text/plain` (Prometheus exposition format)

### Rate Limiting
- Identification header: `X-Client-ID` (decoder) or `X-API-Key` (coordinator)
- Typical error: `429 Too Many Requests`

### GET /loaded_adapters
- Lists adapters currently hot-loaded in GPU memory (LRU cache).
```json
["es", "de_medical"]
```

### POST /compose_adapter (internal)
- Purpose: Compose a zero-shot adapter on the fly (protected for coordinator use).
- Auth: `X-Internal-Auth: <INTERNAL_SERVICE_TOKEN>`
- Body: `{ "source_adapter": "en", "target_adapter": "es", "strategy": "average" }`

### POST /admin/reload_model (admin)
- Auth: `Authorization: Bearer <JWT>` (signed with `DECODER_JWT_SECRET`)
```json
{ "status": "Model reloaded" }
```

### OpenAPI/Docs
- OpenAPI JSON: `/openapi.json`
- Swagger UI: `/docs`

---

## Coordinator API
FastAPI app in `coordinator/advanced_coordinator.py`.

### POST /api/decode
- Purpose: Proxy binary decode requests to the least-loaded healthy decoder.
- Auth: `X-API-Key: <key>` (validated by APIKeyManager)
- Headers:
  - `Content-Type: application/octet-stream`
  - `X-Source-Language: <code>` (required)
  - `X-Target-Language: <code>` (required)
  - `X-Domain: <domain>` (optional)
- Body: Same binary payload as sent to decoder `/decode`.
- Response: Forwards decoder JSON response.

### GET /api/status
- Returns current pool status.
```json
{
  "model_version": "1.0.0",
  "decoder_pool_size": 3,
  "healthy_decoders": 2,
  "decoders": [
    {
      "node_id": "abc123",
      "endpoint": "https://decoder-a.example.com",
      "region": "us-east",
      "gpu_type": "T4",
      "capacity": 64,
      "healthy": true,
      "load": 7,
      "uptime": 12345
    }
  ]
}
```

### Service Discovery and Node Management (v1)
- POST `/api/v1/register` -- Register decoder node
- DELETE `/api/v1/unregister/{node_id}` -- Unregister decoder
- GET `/api/v1/node/{node_id}` -- Get decoder info
- PUT `/api/v1/node/{node_id}/status` -- Update decoder status

### Admin
- POST `/admin/add_decoder` (auth required)
- POST `/admin/remove_decoder` (auth required)
- Login: `POST /login` with `COORDINATOR_TOKEN`
- Dashboard: `GET /` (HTML)

### Error Handling
- `502 Bad Gateway` -- decoder error or unreachable
- `503 Service Unavailable` -- no healthy decoders
- `504 Gateway Timeout` -- upstream timeout

### Metrics
- GET `/metrics` -- Prometheus metrics for the coordinator.

---

## Authentication Summary
- Decoder admin endpoints: `Authorization: Bearer <JWT>` (DECODER_JWT_SECRET)
- Decoder internal `/compose_adapter`: `X-Internal-Auth: <INTERNAL_SERVICE_TOKEN>`
- Coordinator `/api/decode`: `X-API-Key: <key>`
- Coordinator admin UI: `POST /login` with `COORDINATOR_TOKEN`

---

## Error Codes
| Code | Description |
|------|-------------|
| 400  | Invalid payload or headers |
| 401  | Unauthorized (missing/invalid key or token) |
| 413  | Payload too large |
| 422  | Unsupported language pair |
| 429  | Rate limit exceeded |
| 500  | Server error |
| 502  | Upstream decoder error (proxy failure) |
| 503  | No healthy decoders available / model not loaded |

---

## Language Codes
Common ISO codes: `en, es, fr, de, zh, ja, ko, ar, hi, ru, pt, it, tr, th, vi, pl, uk, nl, id, sv`.

---

## Notes
- For web/SDK clients using binary mode, point to coordinator `/api/decode`.
- If deploying a decoder directly, use its `/decode` endpoint.
- OpenAPI docs are available on each service at runtime.
- Binary schema reference: `docs/schemas/binary_payload_schema.md`.

## Related Documentation
- `docs/ONBOARDING.md` (local paths and mounts)
- `docs/DEPLOYMENT.md` (container mounts and defaults)
- `docs/Vocabulary_Guide.md` (vocabulary runtime configuration)
