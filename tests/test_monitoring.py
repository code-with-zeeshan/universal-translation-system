import requests
import pytest

def test_decoder_metrics():
    # Assumes decoder is running and exposes /metrics on port 8000
    resp = requests.get('http://localhost:8000/metrics')
    assert resp.status_code == 200
    assert 'translation_requests_total' in resp.text
    assert 'translation_latency_seconds' in resp.text

def test_system_metrics():
    # Assumes system_metrics.py is running on port 9000
    resp = requests.get('http://localhost:9000')
    assert resp.status_code == 200
    assert 'system_cpu_usage_percent' in resp.text
    assert 'system_ram_usage_percent' in resp.text