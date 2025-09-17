import os
import pytest
from fastapi.testclient import TestClient

# Import decoder FastAPI app
from cloud_decoder.optimized_decoder import app as decoder_app

@pytest.fixture(scope="module")
def client():
    # Minimal env required for security hooks in some setups
    os.environ.setdefault('DECODER_JWT_SECRET', 'test-secret')
    return TestClient(decoder_app)

@pytest.mark.timeout(10)
def test_decode_endpoint_smoke(client):
    # Minimal smoke to ensure endpoint responds
    payload = {
        "source_lang": "en",
        "target_lang": "es",
        "text": "hello world",
    }
    r = client.post("/decode", json=payload)
    assert r.status_code in (200, 202, 400, 422)
    # We accept a range because actual model may be mocked/disabled in tests; success is that the app is alive

@pytest.mark.timeout(10)
def test_health_and_ready(client):
    r1 = client.get("/health")
    r2 = client.get("/ready")
    assert r1.status_code in (200, 204)
    assert r2.status_code in (200, 204)