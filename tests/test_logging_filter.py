import logging
from io import StringIO

import pytest

from utils.logging_config import LoggingSensitiveDataFilter


def _build_test_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove existing handlers to isolate test
    for h in list(logger.handlers):
        logger.removeHandler(h)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(LoggingSensitiveDataFilter())

    assert getattr(rec, "token", None) == LoggingSensitiveDataFilter.MASK

    assert nested_extra.get("password") == LoggingSensitiveDataFilter.MASK

    assert nested_extra["nested"].get("jwt_secret") == LoggingSensitiveDataFilter.MASK