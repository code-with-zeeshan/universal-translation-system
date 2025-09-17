import os
import json
import pytest
from fastapi.testclient import TestClient

# Import decoder app
from cloud_decoder.optimized_decoder import app as decoder_app


@pytest.fixture(autouse=True)
def ensure_test_mode(monkeypatch):
    # Speed up decoder startup for tests
    monkeypatch.setenv("UTS_TEST_MODE", "1")
    # Ensure RS256 is not enforced by default
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    # Clear JWKS_KEYS by re-import side effect is not trivial; rely on /ready logic with rs256 disabled
    yield


def test_decoder_health_ok():
    client = TestClient(decoder_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "apiVersion" in body


def test_decoder_ready_ok_without_rs256():
    client = TestClient(decoder_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)  # model/vocab may be None when UTS_TEST_MODE=1
    body = r.json()
    assert body.get("component") == "decoder"
    assert "checks" in body


def test_decoder_ready_rs256_requires_jwks(monkeypatch):
    # Simulate RS256 configured; no keys loaded -> jwks_ok False => 503
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhki...\n-----END PUBLIC KEY-----")

    client = TestClient(decoder_app)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["jwks"] is False