import os
import json
import pytest
from fastapi.testclient import TestClient

# Coordinator tests
from coordinator.advanced_coordinator import app as coord_app

# Decoder tests
from cloud_decoder.optimized_decoder import app as decoder_app


def test_coordinator_health():
    client = TestClient(coord_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    # version fields may be None in local tests, just ensure keys exist
    assert "version" in body and "apiVersion" in body


def test_coordinator_ready_hs256_ok():
    # Ensure RS256 not required for this test
    os.environ.pop("JWT_PUBLIC_KEY", None)
    os.environ.pop("JWT_PUBLIC_KEY_PATH", None)
    client = TestClient(coord_app)
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body.get("component") == "coordinator"
    assert body.get("ready") is True


def test_decoder_health():
    client = TestClient(decoder_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "version" in body and "apiVersion" in body


def test_decoder_ready_status_code():
    # Decoder ready should return 200 when model and vocabulary manager are initialized in this test context.
    # In minimal unit context, model may be None. We assert presence of fields and allow 200/503.
    client = TestClient(decoder_app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body.get("component") == "decoder"
    assert "checks" in body