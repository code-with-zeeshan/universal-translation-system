from __future__ import annotations

import logging
import sys
import traceback
from functools import wraps
from typing import Any, Callable, Optional

from utils.exceptions import (
    UniversalTranslationError,
    ConfigurationError,
    DataError,
    ModelError,
    SecurityError,
    TrainingError,
    VocabularyError,
)

logger = logging.getLogger(__name__)

_EXIT_CODES = {
    ConfigurationError: 2,
    DataError: 3,
    VocabularyError: 4,
    ModelError: 5,
    TrainingError: 6,
    SecurityError: 7,
    UniversalTranslationError: 1,
}


def run_safely(main_func: Callable[[], Any]) -> int:
    try:
        main_func()
        return 0
    except (ConfigurationError, DataError, VocabularyError,
            ModelError, TrainingError, SecurityError,
            UniversalTranslationError) as exc:
        cls = type(exc)
        exit_code = _EXIT_CODES.get(cls, 1)
        logger.error("[%s] %s", cls.__name__, exc, exc_info=True)
        print(f"\nError [{cls.__name__}]: {exc}", file=sys.stderr)
        return exit_code
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    except Exception as exc:
        logger.critical("Unhandled exception", exc_info=True)
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


def error_boundary(main_func: Callable[..., Any]) -> Callable[..., int]:
    @wraps(main_func)
    def wrapper(*args: Any, **kwargs: Any) -> int:
        return run_safely(lambda: main_func(*args, **kwargs))
    return wrapper
