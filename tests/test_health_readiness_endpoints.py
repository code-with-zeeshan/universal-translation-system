import os
import pytest
from fastapi.testclient import TestClient

from runtime.coordinator.advanced_coordinator import app as coord_app
from runtime.cloud_decoder.optimized_decoder import app as decoder_app


# -- Coordinator endpoint tests --

def test_coordinator_health():
    client = TestClient(coord_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "version" in body and "apiVersion" in body


@pytest.fixture(autouse=True)
def clean_coordinator_env(monkeypatch):
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    monkeypatch.setenv("USE_ETCD", "false")
    yield


def test_coordinator_ready_ok_defaults():
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body.get("component") == "coordinator"
    assert "checks" in body
    assert "redis" in body["checks"]
    assert "etcd" in body["checks"]


def test_coordinator_ready_hs256_ok():
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("component") == "coordinator"
    assert body.get("ready") is True


def test_coordinator_ready_rs256_requires_jwks(monkeypatch):
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhki...\n-----END PUBLIC KEY-----")
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["jwks"] is False


def test_coordinator_ready_with_etcd_disabled(monkeypatch):
    monkeypatch.setenv("USE_ETCD", "false")
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["checks"]["etcd"] in (True, False)


# -- Decoder endpoint tests --

@pytest.fixture(autouse=True)
def ensure_test_mode(monkeypatch):
    monkeypatch.setenv("UTS_TEST_MODE", "1")
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("JWT_PUBLIC_KEY_PATH", raising=False)
    yield


def test_decoder_health():
    client = TestClient(decoder_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "apiVersion" in body


def test_decoder_ready_ok_without_rs256():
    client = TestClient(decoder_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body.get("component") == "decoder"
    assert "checks" in body


def test_decoder_ready_rs256_requires_jwks(monkeypatch):
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhki...\n-----END PUBLIC KEY-----")
    client = TestClient(decoder_app)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["jwks"] is False


def test_metrics_router_mounted_paths_exist():
    client_c = TestClient(coord_app)
    client_d = TestClient(decoder_app)
    rc = client_c.get("/metrics")
    rd = client_d.get("/metrics")
    assert rc.status_code in (401, 200)
    assert rd.status_code in (401, 200)
