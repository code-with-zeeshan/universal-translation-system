import os
import json
import pytest
from utils.jwks_utils import build_jwks_from_env, diff_kids


def test_jwks_build_and_diff_with_spki_kid(monkeypatch):
    # Generate a temporary RSA key pair
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode('utf-8')

    monkeypatch.setenv('JWT_PUBLIC_KEY', pub_pem)
    monkeypatch.delenv('JWT_KEY_IDS', raising=False)

    jwks = build_jwks_from_env(component='test')
    assert len(jwks) == 1
    assert jwks[0]['kty'] == 'RSA'
    assert jwks[0]['alg'] == 'RS256'
    assert 'kid' in jwks[0]

    # Diff with empty should show added kid
    added, removed = diff_kids([], jwks)
    assert added == [jwks[0]['kid']]
    assert removed == []


def test_jwt_tampering_rejected_headers_required(monkeypatch):
    import jwt
    from datetime import datetime, timedelta
    # Set HS256 for test simplicity
    secret = 'super-secret-test'
    monkeypatch.setenv('JWT_PUBLIC_KEY', '')
    # Create valid token
    now = datetime.utcnow()
    claims = {
        'iss': 'test-iss',
        'aud': 'test-aud',
        'iat': int(now.timestamp()),
        'nbf': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=5)).timestamp())
    }
    token = jwt.encode(claims, secret, algorithm='HS256')

    # Tamper: change payload slightly
    parts = token.split('.')
    assert len(parts) == 3
    tampered_payload = parts[1][:-2] + ('A' if parts[1][-2] != 'A' else 'B')
    tampered = parts[0] + '.' + tampered_payload + '.' + parts[2]

    # Validate using PyJWT directly should fail
    with pytest.raises(Exception):
        jwt.decode(
            tampered,
            secret,
            algorithms=['HS256'],
            options={'require': ['exp', 'iat', 'nbf']},
            issuer='test-iss',
            audience='test-aud',
        )


def test_insecure_config_fail_fast(monkeypatch):
    # Decoder must have DECODER_JWT_SECRET
    monkeypatch.delenv('DECODER_JWT_SECRET', raising=False)
    from cloud_decoder import optimized_decoder
    # simulate startup_validation call directly should raise
    with pytest.raises(RuntimeError):
        import asyncio
        asyncio.get_event_loop().run_until_complete(optimized_decoder.startup_validation())


def test_xss_sanitization_dashboard(monkeypatch):
    # Inject a malicious node value and verify it is escaped in the dashboard HTML
    from coordinator.advanced_coordinator import DASHBOARD_TEMPLATE, DecoderNodeSchema
    nodes = [
        DecoderNodeSchema(
            node_id="n1",
            endpoint="http://example.com",
            region="<script>alert(1)</script>",
            gpu_type="T4",
            capacity=1,
            healthy=True,
            load=0,
            uptime=0,
        )
    ]
    html = DASHBOARD_TEMPLATE.render(nodes=nodes, logged_in=False)
    # The script tag should be escaped and not executable in HTML
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html