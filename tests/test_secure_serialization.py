"""Tests for utils.secure_serialization"""
import pytest
from utils.secure_serialization import (
    secure_serialize_json,
    secure_deserialize_json,
    safe_deserialize_json,
    secure_serialize_msgpack,
    secure_deserialize_msgpack,
    safe_deserialize_msgpack,
)


class TestSecureJsonSerialization:
    def test_serialize_deserialize_roundtrip(self):
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "val"}}
        serialized = secure_serialize_json(data)
        assert isinstance(serialized, bytes)
        deserialized = secure_deserialize_json(serialized)
        assert deserialized == data

    def test_serialize_list(self):
        data = [1, "two", 3.0]
        serialized = secure_serialize_json(data)
        deserialized = secure_deserialize_json(serialized)
        assert deserialized == data

    def test_serialize_empty_dict(self):
        data = {}
        serialized = secure_serialize_json(data)
        deserialized = secure_deserialize_json(serialized)
        assert deserialized == {}

    def test_safe_deserialize_json(self):
        data = {"a": 1, "b": "hello"}
        serialized = secure_serialize_json(data)
        result = safe_deserialize_json(serialized)
        assert result == data

    def test_safe_deserialize_invalid(self):
        result = safe_deserialize_json(b"not valid json{{{")
        assert result is None

    def test_deserialize_invalid_raises(self):
        with pytest.raises(Exception):
            secure_deserialize_json(b"garbage{{{")

    def test_serialize_none(self):
        data = None
        serialized = secure_serialize_json(data)
        deserialized = secure_deserialize_json(serialized)
        assert deserialized is None

    def test_large_integer(self):
        """JSON-safe integers should roundtrip."""
        data = {"big": 2**53 - 1}
        serialized = secure_serialize_json(data)
        deserialized = secure_deserialize_json(serialized)
        assert deserialized["big"] == 2**53 - 1


class TestSecureMsgpackSerialization:
    def test_serialize_deserialize_roundtrip(self):
        data = {"name": "test", "count": 42, "flags": [True, False]}
        serialized = secure_serialize_msgpack(data)
        assert isinstance(serialized, bytes)
        deserialized = secure_deserialize_msgpack(serialized)
        assert deserialized == data

    def test_binary_data(self):
        data = {"bytes": b"\x00\x01\x02"}
        serialized = secure_serialize_msgpack(data)
        deserialized = secure_deserialize_msgpack(serialized)
        assert deserialized["bytes"] == b"\x00\x01\x02"

    def test_safe_deserialize_msgpack(self):
        data = [1, 2, 3]
        serialized = secure_serialize_msgpack(data)
        result = safe_deserialize_msgpack(serialized)
        assert result == data

    def test_safe_deserialize_invalid(self):
        result = safe_deserialize_msgpack(b"\xc1\xc2\xc3")
        assert result is None

    def test_deserialize_invalid_raises(self):
        with pytest.raises(Exception):
            secure_deserialize_msgpack(b"\xc1\xc2\xc3")

    def test_none_values(self):
        data = {"key": None}
        serialized = secure_serialize_msgpack(data)
        deserialized = secure_deserialize_msgpack(serialized)
        assert deserialized == data
