import logging
from io import StringIO

import pytest

from utils.logging_config import SensitiveDataFilter


def _build_test_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove existing handlers to isolate test
    for h in list(logger.handlers):
        logger.removeHandler(h)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SensitiveDataFilter())
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, stream


def test_sensitive_filter_masks_message_and_extra(caplog):
    logger, stream = _build_test_logger("test_sensitive")

    with caplog.at_level(logging.INFO):
        logger.info(
            "user login token: abc123 and password: hunter2",
            extra={
                "token": "abc123",
                "extra": {"password": "hunter2", "nested": {"jwt_secret": "top"}},
            },
        )

    # Check formatted output had masked message values
    output = stream.getvalue()
    assert "token: ***" in output.lower()
    assert "password: ***" in output.lower()

    # Check record attributes (structured fields) are masked
    assert len(caplog.records) >= 1
    rec = caplog.records[0]
    # Flat field masked
    assert getattr(rec, "token", None) == SensitiveDataFilter.MASK
    # Nested dict masked for known sensitive keys
    nested_extra = getattr(rec, "extra", {})
    assert isinstance(nested_extra, dict)
    assert nested_extra.get("password") == SensitiveDataFilter.MASK
    assert isinstance(nested_extra.get("nested"), dict)
    assert nested_extra["nested"].get("jwt_secret") == SensitiveDataFilter.MASK