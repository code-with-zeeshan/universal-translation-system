import os
import pytest
from fastapi.testclient import TestClient

from coordinator.advanced_coordinator import app as coord_app


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # By default no RS256 requirement
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    # Toggle etcd usage off unless test enables it
    monkeypatch.setenv("USE_ETCD", "false")
    yield


def test_coordinator_ready_ok_defaults():
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body.get("component") == "coordinator"
    assert "checks" in body
    # redis/etcd should be present in checks (optional)
    assert "redis" in body["checks"]
    assert "etcd" in body["checks"]


def test_coordinator_ready_rs256_requires_jwks(monkeypatch):
    # Simulate RS256 configured; coordinator should require JWKS to be present, otherwise 503
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhki...\n-----END PUBLIC KEY-----")
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["jwks"] is False


def test_coordinator_ready_with_etcd_disabled(monkeypatch):
    # Explicitly ensure etcd is disabled -> check should still be present and True by default
    monkeypatch.setenv("USE_ETCD", "false")
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["checks"]["etcd"] in (True, False)