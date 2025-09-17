import json
from fastapi.testclient import TestClient

# Coordinator app
from coordinator.advanced_coordinator import app as coordinator_app
# Decoder app
from cloud_decoder.optimized_decoder import app as decoder_app


def test_coordinator_health_ready_routes():
    client = TestClient(coordinator_app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data and data["status"] == "ok"
    assert "apiVersion" in data

    r2 = client.get("/ready")
    assert r2.status_code in (200, 503)  # readiness may be false in CI
    payload = r2.json()
    assert "component" in payload and payload["component"] == "coordinator"
    assert "checks" in payload


def test_decoder_health_ready_routes():
    client = TestClient(decoder_app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"

    r2 = client.get("/ready")
    assert r2.status_code in (200, 503)
    payload = r2.json()
    assert payload.get("component") == "decoder"
    assert "checks" in payload


def test_metrics_router_mounted_paths_exist():
    client_c = TestClient(coordinator_app)
    client_d = TestClient(decoder_app)
    # We don't provide JWT in tests; ensure 401 rather than 404
    rc = client_c.get("/metrics")
    rd = client_d.get("/metrics")
    assert rc.status_code in (401, 200)
    assert rd.status_code in (401, 200)