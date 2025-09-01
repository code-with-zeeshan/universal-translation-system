# Binary Payload Schema (Edge → Decoder)

This document describes the structure of the binary payload sent by edge encoders to the decoder `/decode` endpoint.

## Encoding
- Container: MsgPack
- Compression: LZ4 frame
- Typical size: 1–8 KB

## MsgPack Structure
```jsonc
{
  // Optional for pure embedding mode (server-only decode)
  "tokens": [int, ...],            // e.g., subword/token IDs

  // Optional; when present, embeddings must match server expectations
  "embeddings": [float32, ...],   // contiguous float32 array (row-major)

  "metadata": {
    "source_language": "en",     // ISO 639-1/2 code
    "text_hash": "sha1:...",     // useful for dedup/cache
    "domain": "medical",         // optional domain hint
    "client_id": "abc123"        // aligns with X-Client-ID header
  }
}
```

Notes:
- When both `tokens` and `embeddings` are present, the server may prefer embeddings and ignore tokens depending on configuration.
- Fields are extensible; unknown fields should be ignored by the server for forward compatibility.

## Compression
- Use LZ4 frame with default settings.
- On the server side, decompress first, then msgpack-unpack the buffer.

## Example (pseudo-code)
```python
import msgpack, lz4.frame
payload = {
    "tokens": [101, 202, 303],
    "metadata": {"source_language": "en", "text_hash": "sha1:...."}
}
bin_msgpack = msgpack.packb(payload, use_bin_type=True)
compressed = lz4.frame.compress(bin_msgpack)
# POST compressed to /decode with Content-Type: application/octet-stream
```