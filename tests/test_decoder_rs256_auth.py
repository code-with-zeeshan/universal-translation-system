# tests/test_decoder_rs256_auth.py
import os
import importlib
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


def gen_rsa_pair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return priv_pem, pub_pem


def test_require_jwt_rs256_accepts_valid_token(monkeypatch):
    # Strong secrets required by bootstrap/validation
    os.environ["DECODER_JWT_SECRET"] = "x" * 40
    os.environ["INTERNAL_SERVICE_TOKEN"] = "y" * 40
    os.environ["UTS_HMAC_KEY"] = "z" * 40

    priv_pem, pub_pem = gen_rsa_pair()

    os.environ["JWT_PUBLIC_KEY"] = pub_pem
    os.environ["JWT_ISS"] = "uts"
    os.environ["JWT_AUD"] = "decoder"

    # Create RS256 token
    payload = {
        "sub": "tester",
        "iss": "uts",
        "aud": "decoder",
        "iat": int(datetime.now(tz=timezone.utc).timestamp()),
        "nbf": int(datetime.now(tz=timezone.utc).timestamp()),
        "exp": int((datetime.now(tz=timezone.utc) + timedelta(minutes=5)).timestamp()),
    }
    token = jwt.encode(payload, priv_pem, algorithm="RS256")

    # Import module fresh to pick env vars
    if "cloud_decoder.optimized_decoder" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("cloud_decoder.optimized_decoder"))
    mod = importlib.import_module("cloud_decoder.optimized_decoder")

    # Build credentials object expected by dependency
    creds = SimpleNamespace(credentials=token)

    # Should not raise
    mod.require_jwt(creds)