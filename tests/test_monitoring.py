import os
import pytest
import requests


def test_decoder_metrics():
    # Optional e2e test: only run if RUN_E2E_TESTS=1 and endpoint provided
    if os.environ.get('RUN_E2E_TESTS') != '1':
        pytest.skip('Skipping e2e metrics test; set RUN_E2E_TESTS=1 to enable')
    endpoint = os.environ.get('DECODER_METRICS_ENDPOINT', 'http://localhost:8000/metrics')
    try:
        resp = requests.get(endpoint, timeout=5)
    except Exception:
        pytest.skip('Decoder metrics endpoint not available')
    assert resp.status_code == 200
    assert 'coordinator_decoder_active' in resp.text or 'translation_requests_total' in resp.text


def test_system_metrics():
    # Optional: only run if endpoint provided
    endpoint = os.environ.get('SYSTEM_METRICS_ENDPOINT', '')
    if not endpoint:
        pytest.skip('SYSTEM_METRICS_ENDPOINT not provided')
    try:
        resp = requests.get(endpoint, timeout=5)
    except Exception:
        pytest.skip('System metrics endpoint not available')
    assert resp.status_code == 200