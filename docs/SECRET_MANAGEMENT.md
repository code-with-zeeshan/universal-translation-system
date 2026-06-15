# Secret Management

Three-layer architecture for secure credential storage, bootstrap, serialization, and rotation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Bootstrap (secrets_bootstrap.py)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  VAR_FILE → read file → set VAR in environ        │  │
│  │  validate_runtime_secrets(role) → fail fast       │  │
│  │  rotate_secret_if_expired() → auto-rotation       │  │
│  └───────────────────────────────────────────────────┘  │
│                           │                              │
│                           ▼                              │
│  Layer 2: Credential Manager (credential_manager.py)     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. Environment variable (UTS_<NAME>)             │  │
│  │  2. System keyring (keyring.get_password)         │  │
│  │  3. Encrypted file (~/.uts/credentials.json)      │  │
│  └───────────────────────────────────────────────────┘  │
│                           │                              │
│                           ▼                              │
│  Layer 3: Secure Serialization (secure_serialization.py)│
│  ┌───────────────────────────────────────────────────┐  │
│  │  HMAC-SHA256 signed JSON / MsgPack                │  │
│  │  Type validation, size limits, restricted pickle  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Bootstrap (`utils/secrets_bootstrap.py`)

### File-Based Secrets (Recommended for Docker/K8s)

The `*_FILE` pattern reads secrets from mounted files instead of env vars:

```
COORDINATOR_JWT_SECRET_FILE=/run/secrets/coordinator_jwt_secret
```

On startup, `bootstrap_secrets(role)` iterates all `FILE_ENV_PAIRS`. If a `*_FILE` is set and the file exists, the content is read into the corresponding `*` env var. File permissions are validated (0600 recommended).

### Supported File Pairs

| File Env Var | Target Env Var |
|---|---|
| `COORDINATOR_SECRET_FILE` | `COORDINATOR_SECRET` |
| `COORDINATOR_JWT_SECRET_FILE` | `COORDINATOR_JWT_SECRET` |
| `COORDINATOR_TOKEN_FILE` | `COORDINATOR_TOKEN` |
| `INTERNAL_SERVICE_TOKEN_FILE` | `INTERNAL_SERVICE_TOKEN` |
| `DECODER_JWT_SECRET_FILE` | `DECODER_JWT_SECRET` |
| `JWT_PRIVATE_KEY_FILE` | `JWT_PRIVATE_KEY` |
| `REDIS_PASSWORD_FILE` | `REDIS_PASSWORD` |
| `HF_TOKEN_FILE` | `HF_TOKEN` |

### Runtime Validation

`validate_runtime_secrets(role)` checks:

| Role | Checks |
|---|---|
| `general` | `UTS_HMAC_KEY` ≥32 chars |
| `coordinator` | + `COORDINATOR_SECRET`, `COORDINATOR_TOKEN`, `INTERNAL_SERVICE_TOKEN` all ≥32 chars. Either `COORDINATOR_JWT_SECRET` (HS256) ≥32 chars OR RS256 keypair (≥2048 bits) |
| `decoder` | + `DECODER_JWT_SECRET` ≥32 chars, `INTERNAL_SERVICE_TOKEN` ≥32 chars |

Placeholder values (like `use-openssl-rand-hex-32-to-generate-a-secure-key`) are rejected.

### Expiry Tracking

Each secret can have a companion `*_EXPIRY` env var (Unix timestamp). `rotate_secret_if_expired()` checks expiry and auto-rotates via the credential manager. Default rotation period: 90 days.

## Layer 2: Credential Manager (`utils/credential_manager.py`)

### Lookup Order

```
1. Environment variable:       UTS_<NAME> (e.g., UTS_HMAC_KEY)
2. System keyring:            keyring.get_password("UniversalTranslationSystem", key)
3. Encrypted file:            ~/.universaltranslationsystem/credentials.json
```

### Storage Methods

| Method | Where | Security |
|---|---|---|
| Environment | `os.environ` | Depends on deployment |
| Keyring (`store_in="keyring"`) | OS keychain (macOS Keychain, KDE Wallet, etc.) | Strong — OS-managed encryption |
| Encrypted file (`store_in="config"`) | `~/.universaltranslationsystem/credentials.json` | PBKDF2-derived Fernet key (100K iterations, SHA-256) |

### Key Features

- Thread-safe (RLock-protected in-memory cache)
- Singleton `credential_manager` instance for app-wide use
- `get()`, `set()`, `delete()`, `list_keys()`, `clear()`
- Auto-fallback: keyring → file if keyring unavailable

## Layer 3: Secure Serialization (`utils/secure_serialization.py`)

### HMAC-Signed Formats

| Function | Format | Use Case |
|---|---|---|
| `secure_serialize_json` / `secure_deserialize_json` | `HMAC:SHA256` : `JSON` | Cross-service payloads |
| `secure_serialize_msgpack` / `secure_deserialize_msgpack` | `HMAC:SHA256` + `MsgPack` | Binary payloads (LZ4-compressed) |
| `secure_serialize_json_compressed` / `secure_deserialize_json_compressed` | `HMAC:SHA256` : `base64(zlib(JSON))` | Large payloads |
| `secure_serialize_with_version` / `secure_deserialize_with_version` | Version-wrapped + HMAC | Versioned data |
| `secure_deserialize_with_schema` | JSON Schema validation + HMAC | Schema-enforced data |

### Security Controls

- **Size limit:** 100MB max deserialization
- **Type validation:** Only allows `None`, `bool`, `int`, `float`, `str`, `bytes`, `list`, `tuple`, `dict`, `set`
- **Restricted pickle:** Allow-listed modules only
- **HMAC key:** Must be `UTS_HMAC_KEY` (≥32 chars)

## API Key Management (`utils/auth.py`)

`APIKeyManager` manages client API keys for the coordinator:

- **Secure storage:** Keys stored as PBKDF2-HMAC-SHA256 hashes (600K iterations)
- **Metadata tracking:** Client name, permissions, creation date, last used, request count
- **Rotation:** `rotate_api_key()` revokes old key and issues new one for same client
- **Encrypted backend:** Set `UTS_API_KEYS_USE_CREDMGR=true` to store metadata via CredentialManager instead of plaintext JSON

## Secret Rotation (`tools/rotate_secrets.py`)

```bash
# Rotate all 6 key secrets (generates new random values, stores via credential manager)
uts tools --rotate-secrets

# Rotate a specific secret
uts tools --rotate-secrets --key-name coordinator_jwt_secret

# Generate RS256 keypair
uts tools --rotate-secrets --type rs256 --kid key-1

# Preview without persisting (sets in current env)
uts tools --rotate-secrets --key-name coordinator_jwt_secret --set-env
```

### Rotation Workflow

1. `rotate_hs256(key_name)` generates `secrets.token_urlsafe(48)`, stores via credential manager, prints the `_EXPIRY` var
2. `generate_rs256_pair()` creates a new RSA 2048-bit keypair
3. For RS256: add new public key to `JWT_PUBLIC_KEY_PATH`, keep old keys for grace period, then remove
4. Set the printed `_EXPIRY` in your deployment to enable auto-rotation monitoring

## Best Practices

1. **Never commit secrets** to version control — use `.env` (gitignored), Docker Secrets, or K8s Secrets
2. **Prefer `*_FILE` pattern** in containers — avoids secrets in env var listings
3. **Rotate every 90 days** — `uts tools --rotate-secrets` handles this
4. **Set `UTS_ROLE`** — enables targeted validation (`general`, `coordinator`, `decoder`)
5. **RS256 for multi-service** — use asymmetric keys when decoder and coordinator run on separate hosts
6. **File permissions 0600** for all mounted secret files
7. **Validate at startup** — `validate_runtime_secrets()` fails fast with clear error messages

## Related Documentation

- `docs/environment-variables.md` — Complete env var reference
- `docs/SECURITY_BEST_PRACTICES.md` — Broader security guidelines
- `docs/API.md` — Auth summary for API endpoints
