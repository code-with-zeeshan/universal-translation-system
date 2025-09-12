# tests/test_jwks_reload_stub.py
# This is a lightweight stub to exercise JWKS utilities without spinning the server.

from utils.jwks_utils import build_jwks_from_env, diff_kids
import os


def test_build_jwks_handles_empty_env():
    # No keys -> empty list
    os.environ.pop("JWT_PUBLIC_KEY", None)
    os.environ.pop("JWT_PUBLIC_KEY_PATH", None)
    jwks = build_jwks_from_env(component="test")
    assert isinstance(jwks, list)


def test_diff_kids_changeset():
    old = [{"kid": "a"}, {"kid": "b"}]
    new = [{"kid": "b"}, {"kid": "c"}]
    added, removed = diff_kids(old, new)
    assert added == ["c"]
    assert removed == ["a"]