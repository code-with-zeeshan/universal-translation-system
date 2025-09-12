# tests/test_coordinator_rs256_validation.py
import os
import pytest

from utils.secrets_bootstrap import validate_runtime_secrets


@pytest.fixture
def cleanup_env():
    keys = [
        "COORDINATOR_SECRET",
        "COORDINATOR_TOKEN",
        "INTERNAL_SERVICE_TOKEN",
        "COORDINATOR_JWT_SECRET",
        "JWT_PRIVATE_KEY",
        "JWT_PUBLIC_KEY",
        "UTS_HMAC_KEY",
    ]
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_coordinator_rs256_allows_missing_hs_secret(cleanup_env):
    # Required core secrets
    os.environ["COORDINATOR_SECRET"] = "a" * 40
    os.environ["COORDINATOR_TOKEN"] = "b" * 40
    os.environ["INTERNAL_SERVICE_TOKEN"] = "c" * 40
    os.environ["UTS_HMAC_KEY"] = "h" * 40
    # Provide RS256 keys only
    os.environ["JWT_PRIVATE_KEY"] = """-----BEGIN PRIVATE KEY-----\nMIIBVwIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEAw3q/7mP5S6k0p9s7\nQy+Rj9J3rQXhQ8iKp0r7bik0j3LZ0+f9eY0yS8z8iR6Q8S0xU0qfVbR2i1sWg2z3\nGQIDAQABAkAEV7oC5gqWl+Kk2u9XzvJ8O+HhFh0wRj2H7W5E+ZpJ0o9G+JgKl8mS\n+qP6UKl6qE9D1m3mGdWwq8o3vPpU6yEBAiEA7fC1j9FqM7S1bqj8N8vXcFq1w4vX\n6a2K8r9eZb8oV8ECAwEAAQ==\n-----END PRIVATE KEY-----"""
    os.environ["JWT_PUBLIC_KEY"] = """-----BEGIN PUBLIC KEY-----\nMFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIN4kVhVZ4+ZcIYkZ4z0Ej3ITeXy0pPZ\nFQ4yG8Wv5xVdQZJ9VL7YHfQ0o5Kf7H8DgC1s7zNvC4fS7m7GOPoTtUcCAwEAAQ==\n-----END PUBLIC KEY-----"""
    # The dummy PEM above won't parse for real; validator only checks presence/length and HS/RS choice
    validate_runtime_secrets(role="coordinator")


def test_coordinator_rs256_rejects_weak_core_secrets(cleanup_env):
    os.environ["COORDINATOR_SECRET"] = "short"
    os.environ["COORDINATOR_TOKEN"] = "short"
    os.environ["INTERNAL_SERVICE_TOKEN"] = "short"
    os.environ["UTS_HMAC_KEY"] = "h" * 40
    os.environ["JWT_PRIVATE_KEY"] = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
    os.environ["JWT_PUBLIC_KEY"] = "-----BEGIN PUBLIC KEY-----\nabc\n-----END PUBLIC KEY-----"

    with pytest.raises(Exception) as ex:
        validate_runtime_secrets(role="coordinator")
    msg = str(ex.value)
    assert "COORDINATOR_SECRET" in msg or "COORDINATOR_TOKEN" in msg or "INTERNAL_SERVICE_TOKEN" in msg