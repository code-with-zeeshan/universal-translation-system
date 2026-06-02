# utils/logging_config.py
import logging
import logging.config
import os
from pathlib import Path
import sys

# Ensure standard logs folder structure exists
try:
    from utils.common_utils import DirectoryManager
except Exception:
    DirectoryManager = None  # Avoid import cycles during bootstrap

from utils.constants import LOG_DIR

def _get_memory_info() -> str:
    """Get memory information"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"{memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total"
    except ImportError:
        return "Memory info unavailable (psutil not installed)"

def _log_system_info(logger: logging.Logger):
    """Log standardized system information"""
    logger.info("=" * 60)
    logger.info("SYSTEM INFO")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Available memory: {_get_memory_info()}")
    logger.info("=" * 60)

class LoggingSensitiveDataFilter(logging.Filter):
    """Mask common sensitive values in log messages and structured fields.
    - Scrubs message strings via regex (best-effort)
    - Scrubs record attributes added via `extra={}` (structured fields)
    """
    SENSITIVE_KEYS = (
        'authorization', 'proxy-authorization', 'x-api-key', 'api_key', 'apikey',
        'token', 'access_token', 'refresh_token', 'secret', 'password', 'jwt',
        'jwt_secret', 'coordinator_token', 'internal_service_token'
    )
    MASK = '***'

    def _mask_value(self, key: str, value):
        try:
            if value is None:
                return value
            # Strings: mask fully
            if isinstance(value, str):
                return self.MASK
            # Dicts: shallow-mask by key
            if isinstance(value, dict):
                return {k: (self.MASK if str(k).lower() in self.SENSITIVE_KEYS else v) for k, v in value.items()}
            # Sequences: mask string-like elements
            if isinstance(value, (list, tuple)):
                return [self.MASK if isinstance(v, str) else v for v in value]
            # Fallback: mask
            return self.MASK
        except Exception:
            return self.MASK

    def filter(self, record: logging.LogRecord) -> bool:
        import re
        # 1) Scrub message content (best-effort)
        msg = str(record.getMessage())
        for key in self.SENSITIVE_KEYS:
            patterns = [
                rf'(?i)({key})\s*[:=]\s*[^,\s\"]+',
                rf'(?i)"({key})"\s*:\s*"[^"]*"',
                rf'(?i)\b({key})\b\s*=>\s*[^,\s\"]+',
            ]
            for pat in patterns:
                msg = re.sub(
                    pat,
                    lambda m: re.sub(r'(:|=|=>).*', lambda _: f"{m.group(1)}: {self.MASK}", m.group(0), count=1),
                    msg,
                )
        record.msg = msg

        # 2) Scrub structured fields added via LoggerAdapter/extra
        try:
            rec_dict = record.__dict__
            for k in list(rec_dict.keys()):
                if isinstance(k, str) and k.lower() in self.SENSITIVE_KEYS:
                    rec_dict[k] = self._mask_value(k, rec_dict.get(k))
            # Common nested containers
            for container_key in ("extra", "context", "payload"):
                if container_key in rec_dict and isinstance(rec_dict[container_key], dict):
                    nested = rec_dict[container_key]
                    for nk in list(nested.keys()):
                        if isinstance(nk, str) and nk.lower() in self.SENSITIVE_KEYS:
                            nested[nk] = self._mask_value(nk, nested.get(nk))
        except Exception:
            # Never break logging
            pass
        return True

class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Enforces structured logging with `extra` context and avoids string interpolation of secrets.
    Usage: logger.info("event", extra={"user_id": uid, "action": action})
    """
    def process(self, msg, kwargs):
        # Ensure `extra` exists and is a dict
        extra = kwargs.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {"extra": str(extra)}
        # Attach adapter context under `context` and merge flat extras
        base = dict(self.extra) if isinstance(self.extra, dict) else {}
        # Provide a namespaced context to avoid collisions
        combined_extra = {**extra}
        if base:
            combined_extra["context"] = {**base, **combined_extra.get("context", {})}
        kwargs["extra"] = combined_extra
        return msg, kwargs

def get_logger(name: str, context: dict | None = None) -> StructuredLoggerAdapter:
    """Return a structured logger adapter bound with optional context."""
    return StructuredLoggerAdapter(logging.getLogger(name), context or {})

_logging_initialized = False

def _is_worker_process() -> bool:
    """Detect if running in a child process (DataLoader worker, etc.)"""
    try:
        import multiprocessing
        return multiprocessing.parent_process() is not None
    except Exception:
        return False

def setup_logging(log_dir: str = LOG_DIR, log_level: str = "INFO"):
    """Setup comprehensive logging configuration (idempotent, skips DataLoader workers)"""
    global _logging_initialized
    if _logging_initialized:
        return
    # Skip in worker processes — inherits parent logging
    if _is_worker_process():
        _logging_initialized = True  # Prevent repeated checks
        return
    _logging_initialized = True
    # Create log directory and standard sections
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        if DirectoryManager:
            DirectoryManager.create_logs_structure("logs")
    except Exception:
        # Non-fatal; logging will still proceed
        pass

    # Choose formatter dynamically
    use_json = os.getenv('LOG_FORMAT', '').lower() == 'json'

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s'
            }
        },
        'filters': {
            'sensitive': {
                '()': LoggingSensitiveDataFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'json' if use_json else 'standard',
                'filters': ['sensitive'],
                'stream': sys.stdout
            },
            # Root/system files
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/translation_system.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/errors.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            # Section-specific files (always under logs/<section>)
            'file_training': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/training/training.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_data': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/data/data.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_monitoring': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/monitoring/monitoring.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_coordinator': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/coordinator/coordinator.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_decoder': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/decoder/decoder.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_vocabulary': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{LOG_DIR}/vocabulary/vocabulary.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
        },
        'loggers': {
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_training'],
                'propagate': False
            },
            'data': {
                'level': 'INFO',
                'handlers': ['console', 'file_data'],
                'propagate': False
            },
            'monitoring': {
                'level': 'INFO',
                'handlers': ['console', 'file_monitoring'],
                'propagate': False
            },
            'coordinator': {
                'level': 'INFO',
                'handlers': ['console', 'file_coordinator'],
                'propagate': False
            },
            'decoder': {
                'level': 'INFO',
                'handlers': ['console', 'file_decoder'],
                'propagate': False
            },
            'vocabulary': {
                'level': 'INFO',
                'handlers': ['console', 'file_vocabulary'],
                'propagate': False
            },
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file', 'error_file']
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)

    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Universal Translation System - Logging Initialized")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log level: {log_level}")
    logger.info("="*60)

    # Log system info
    _log_system_info(logger)