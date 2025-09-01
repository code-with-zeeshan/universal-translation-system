# API Documentation

## Overview
Two public surfaces exist:
- Decoder Node API (GPU node serving translations)
- Coordinator API (routes requests to least‑loaded healthy decoder nodes)

Use binary `/decode` for highest performance. A JSON `/translate` endpoint is optional and not enabled in the current codebase.

---

## Base URLs
- Decoder Node: http(s)://<decoder-host>:<port>
- Coordinator: http(s)://<coordinator-host>:<port>

---

## Decoder Node API
FastAPI app in `cloud_decoder/optimized_decoder.py`.

### POST /decode
- Purpose: Decode locally encoded embeddings (binary) to a translation.
- Headers:
  - `Content-Type: application/octet-stream`
  - `X-Target-Language: <code>` (required)
  - `X-Domain: <domain>` (optional, e.g., medical, legal)
  - `X-Client-ID: <id>` (optional; used for rate limiting)
- Body: Binary payload produced by the edge encoder, typically LZ4 + MsgPack compressed.
- Response: `application/json`
```json
{
  "translation": "Hola mundo",
  "target_language": "es"
}
```
- Notes:
  - The decoder decompresses and runs generation on GPU.
  - Some deployments may expose this as the root path `/`.

Binary Payload Format:
- Encoded as `lz4`-compressed `msgpack`.
- Typical msgpack fields:
  - `tokens`: int[] (token IDs or subword indices)
  - `embeddings`: float32[] or omitted if server-only decode
  - `metadata`: { `source_language`: string, `text_hash`: string, ... }
- Size: usually a few KB per request.

Examples:
- curl (binary file):
```bash
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  -H "X-Target-Language: es" \
  --data-binary @compressed_embeddings.bin \
  https://decoder.example.com/decode
```
- JavaScript fetch (with Uint8Array):
```ts
const res = await fetch('https://decoder.example.com/decode', {
  method: 'POST',
  headers: { 'Content-Type': 'application/octet-stream', 'X-Target-Language': 'es' },
  body: encodedUint8Array,
});
const data = await res.json();
```

### GET /health
- Returns basic health info.
- Response:
```json
{ "status": "healthy", "device": "cuda" }
```

### GET /status
- Returns extended status and metrics summary.
- Response (example):
```json
{
  "model_version": "1.0.0",
  "healthy": true,
  "metrics": { /* Prometheus summary snapshot */ },
  "vocabulary": { /* vocabulary stats */ }
}
```

### GET /metrics
- Prometheus metrics endpoint.
- Response: `text/plain` (Prometheus exposition format)

### Rate Limiting
- Identification header: `X-Client-ID` (decoder) or `X-API-Key` (coordinator)
- Typical error when exceeded: `429 Too Many Requests`
- Suggested response headers (if enabled in deployment):
  - `X-RateLimit-Limit-Minute`: integer
  - `X-RateLimit-Remaining-Minute`: integer
  - `Retry-After`: seconds

### GET /loaded_adapters
- Lists adapters currently hot‑loaded in GPU memory (LRU cache).
- Response (example):
```json
["es", "de_medical"]
```

### POST /compose_adapter (internal)
- Purpose: Compose a zero‑shot adapter on the fly (protected for coordinator use).
- Auth: `X-Internal-Auth: <INTERNAL_SERVICE_TOKEN>`
- Body:
```json
{ "source_adapter": "en", "target_adapter": "es", "strategy": "average" }
```
- Response:
```json
{ "status": "success", "composed_adapter_name": "es" }
```

### POST /admin/reload_model (admin)
- Auth: `Authorization: Bearer <JWT>` (signed with `DECODER_JWT_SECRET`)
- Response:
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
- Purpose: Proxy binary decode requests to the least‑loaded healthy decoder.
- Auth: `X-API-Key: <key>` (validated by APIKeyManager)
- Headers:
  - `Content-Type: application/octet-stream`
  - `X-Source-Language: <code>` (required)
  - `X-Target-Language: <code>` (required)
  - `X-Domain: <domain>` (optional)
- Body: Same binary payload as sent to decoder `/decode`.
- Response: Forwards decoder JSON response.
- Behavior:
  - If the language pair is unsupported directly, the coordinator may perform zero‑shot pivot via adapter composition and then proxy to the selected node.

Examples:
- curl:
```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Source-Language: en" \
  -H "X-Target-Language: es" \
  --data-binary @compressed_embeddings.bin \
  https://coord.example.com/api/decode
```
- JavaScript fetch:
```ts
const res = await fetch('https://coord.example.com/api/decode', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/octet-stream',
    'X-API-Key': process.env.API_KEY!,
    'X-Source-Language': 'en',
    'X-Target-Language': 'es',
  },
  body: encodedUint8Array,
});
const data = await res.json();
```

### GET /api/status
- Returns current pool status.
- Response (example):
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
- POST `/api/v1/register`
  - Body: DecoderNodeSchema
  - Response: `{ "success": true, "node_id": "..." }`
- DELETE `/api/v1/unregister/{node_id}`
  - Response: `{ "success": true }`
- GET `/api/v1/node/{node_id}`
  - Response: DecoderNodeSchema
- PUT `/api/v1/node/{node_id}/status`
  - Form fields: `healthy: bool`, `load: int`
  - Response: `{ "success": true }`

DecoderNodeSchema:
```json
{
  "node_id": "string",
  "endpoint": "https://decoder.example.com",
  "region": "us-east",
  "gpu_type": "T4|A100|...",
  "capacity": 64,
  "healthy": true,
  "load": 0,
  "uptime": 0
}
```

### Admin
- POST `/admin/add_decoder` (auth required) — JSON body: DecoderNodeSchema
- POST `/admin/remove_decoder` (auth required) — Form field: `node_id`
- Login: `POST /login` with form field `token` (compared to env `COORDINATOR_TOKEN`); sets a session cookie.
- Logout: `POST /logout`
- Dashboard: `GET /` (HTML)

### Error Handling & Upstream Behaviors
- When proxying to a decoder, the coordinator may return:
  - `502 Bad Gateway` if the decoder returns an error or is unreachable
  - `503 Service Unavailable` if no healthy decoders are available
  - `504 Gateway Timeout` if upstream exceeds timeout (deployment dependent)
- Error payload (typical):
```json
{ "error": "string", "details": "optional string", "code": 502 }
```

### Metrics
- GET `/metrics` — Prometheus metrics for the coordinator.

---

## Authentication Summary
- Decoder admin endpoints: `Authorization: Bearer <JWT>` signed with `DECODER_JWT_SECRET`.
- Decoder internal endpoint `/compose_adapter`: `X-Internal-Auth: <INTERNAL_SERVICE_TOKEN>`.
- Coordinator `/api/decode`: `X-API-Key: <key>`.
- Coordinator admin UI: `POST /login` with `COORDINATOR_TOKEN` sets session cookie.

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
Common ISO codes used by headers and payloads (non‑exhaustive): `en, es, fr, de, zh, ja, ko, ar, hi, ru, pt, it, tr, th, vi, pl, uk, nl, id, sv`.

---

## Notes
- For web/SDK clients using binary mode, point to coordinator `/api/decode` for automatic load balancing.
- If you deploy a decoder directly to clients, use its `/decode` endpoint.
- OpenAPI docs are available on each service for introspection at runtime.
- Binary schema reference: see `docs/schemas/binary_payload_schema.md`.