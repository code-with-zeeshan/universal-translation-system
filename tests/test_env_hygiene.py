# tests/test_env_hygiene.py
from pathlib import Path


def test_env_example_mentions_file_variants():
    text = Path('.env.example').read_text(encoding='utf-8')
    assert 'DECODER_JWT_SECRET_FILE' in text
    assert 'COORDINATOR_SECRET_FILE' in text
    assert 'COORDINATOR_JWT_SECRET_FILE' in text
    assert 'COORDINATOR_TOKEN_FILE' in text
    assert 'INTERNAL_SERVICE_TOKEN_FILE' in text


def test_env_example_no_insecure_defaults_in_assignments():
    text = Path('.env.example').read_text(encoding='utf-8')
    # ensure insecure defaults are not used as real values besides explicit "use-openssl..." placeholders for guidance
    assert 'jwtsecret123' not in text
    assert 'a-very-secret-key-for-cookies' not in text
    assert 'a-super-secret-jwt-key' not in text
    assert 'internal-secret-token-for-service-auth' not in text