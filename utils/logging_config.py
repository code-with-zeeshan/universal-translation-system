from utils.common_utils import RuntimeDirectoryManager
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
_MAIN_PID = os.getpid()

def _is_worker_process() -> bool:
    """Detect DataLoader worker processes (works with fork and spawn).
    
    For fork workers: PID comparison catches child processes.
    For spawn workers: the OP_MAIN_PID env var (set in launch.py) won't match os.getpid()."""
    if os.getpid() != _MAIN_PID:
        return True
    main_pid = os.environ.get('OP_MAIN_PID')
    if main_pid and os.getpid() != int(main_pid):
        return True
    return False

def setup_logging(log_dir: str = str(RuntimeDirectoryManager().logs_dir), log_level: str = "INFO"):
    """Setup comprehensive logging configuration (idempotent, skips DataLoader workers)"""
    global _logging_initialized
    if _logging_initialized or _is_worker_process():
        return
    _logging_initialized = True
    # Create log directory and standard sections
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        if DirectoryManager:
            DirectoryManager.create_logs_structure(log_dir)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug("logs_subdirectory_creation_failed", exc_info=True)

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
                'filename': f'{log_dir}/universal_translation_system.log',
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
                'filename': f'{log_dir}/training/training.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_training_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/training/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_data': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/data/data.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_data_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/data/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_monitoring': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/monitoring/monitoring.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_monitoring_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/monitoring/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_coordinator': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/coordinator/coordinator.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_coordinator_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/coordinator/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_decoder': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/decoder/decoder.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_decoder_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/decoder/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_vocabulary': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/vocabulary/vocabulary.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_vocabulary_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/vocabulary/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_evaluation': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/evaluation/evaluation.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_evaluation_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if use_json else 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/evaluation/error.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
        },
        'loggers': {
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_training', 'file_training_error', 'error_file'],
                'propagate': False
            },
            'data': {
                'level': 'INFO',
                'handlers': ['console', 'file_data', 'file_data_error', 'error_file'],
                'propagate': False
            },
            'monitoring': {
                'level': 'INFO',
                'handlers': ['console', 'file_monitoring', 'file_monitoring_error', 'error_file'],
                'propagate': False
            },
            'coordinator': {
                'level': 'INFO',
                'handlers': ['console', 'file_coordinator', 'file_coordinator_error', 'error_file'],
                'propagate': False
            },
            'decoder': {
                'level': 'INFO',
                'handlers': ['console', 'file_decoder', 'file_decoder_error', 'error_file'],
                'propagate': False
            },
            'vocabulary': {
                'level': 'INFO',
                'handlers': ['console', 'file_vocabulary', 'file_vocabulary_error', 'error_file'],
                'propagate': False
            },
            'evaluation': {
                'level': 'INFO',
                'handlers': ['console', 'file_evaluation', 'file_evaluation_error', 'error_file'],
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