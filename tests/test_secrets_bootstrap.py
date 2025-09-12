# tests/test_secrets_bootstrap.py
import os
import tempfile
from utils.secrets_bootstrap import bootstrap_secrets, get_secret, is_strong_secret, validate_runtime_secrets


def test_bootstrap_reads_file_env(tmp_path):
    secret_file = tmp_path / "decoder_secret.txt"
    secret_file.write_text("a" * 40, encoding="utf-8")
    os.environ.pop("DECODER_JWT_SECRET", None)
    os.environ["DECODER_JWT_SECRET_FILE"] = str(secret_file)

    bootstrap_secrets(role="decoder")
    assert os.environ.get("DECODER_JWT_SECRET") == "a" * 40


def test_get_secret_prefers_env_mapping():
    os.environ["INTERNAL_SERVICE_TOKEN"] = "b" * 40
    assert get_secret("INTERNAL_SERVICE_TOKEN") == "b" * 40


def test_is_strong_secret():
    assert not is_strong_secret(None)
    assert not is_strong_secret("short")
    assert is_strong_secret("c" * 40)


def test_validate_runtime_decoder_ok(tmp_path):
    os.environ["DECODER_JWT_SECRET"] = "d" * 40
    os.environ["INTERNAL_SERVICE_TOKEN"] = "e" * 40
    # HMAC is recommended everywhere
    os.environ["UTS_HMAC_KEY"] = "f" * 40
    validate_runtime_secrets(role="decoder")


def test_validate_runtime_decoder_fails():
    os.environ.pop("DECODER_JWT_SECRET", None)
    os.environ.pop("INTERNAL_SERVICE_TOKEN", None)
    os.environ["UTS_HMAC_KEY"] = "g" * 40
    failed = False
    try:
        validate_runtime_secrets(role="decoder")
    except Exception as e:
        failed = True
        assert "DECODER_JWT_SECRET" in str(e) or "INTERNAL_SERVICE_TOKEN" in str(e)
    assert failed