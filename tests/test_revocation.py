import jwt
import pytest
from datetime import datetime, timedelta


def test_revocation_blocked(monkeypatch):
    # Configure HS256 for simplicity in test
    secret = 'test-secret-123456789012345678901234567890'
    monkeypatch.setenv('COORDINATOR_JWT_SECRET', secret)

    # Issue token with jti
    claims = {
        'sub': 'user',
        'jti': 'abc-123',
        'iss': 'test-iss',
        'aud': 'test-aud',
        'iat': int(datetime.utcnow().timestamp()),
        'nbf': int(datetime.utcnow().timestamp()),
        'exp': int((datetime.utcnow() + timedelta(minutes=5)).timestamp())
    }
    token = jwt.encode(claims, secret, algorithm='HS256')

    # Simulate revocation in Redis set
    from coordinator.advanced_coordinator import RedisManager
    rm = RedisManager.get_instance()
    client = rm.get_client()
    if client:
        client.sadd('revoked_jti', 'abc-123')

    # Call require_jwt and expect 401
    from fastapi import HTTPException
    from types import SimpleNamespace
    from coordinator.advanced_coordinator import require_jwt

    class Creds:
        def __init__(self, t):
            self.credentials = t

    with pytest.raises(HTTPException) as ex:
        require_jwt(Creds(token))
    assert ex.value.status_code == 401