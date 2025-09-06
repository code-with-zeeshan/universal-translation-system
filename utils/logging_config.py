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

class SensitiveDataFilter(logging.Filter):
    """Mask common sensitive values in log messages."""
    SENSITIVE_KEYS = (
        'authorization', 'proxy-authorization', 'x-api-key', 'api_key', 'apikey',
        'token', 'access_token', 'refresh_token', 'secret', 'password', 'jwt',
        'jwt_secret', 'coordinator_token', 'internal_service_token'
    )
    MASK = '***'

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        lowered = msg.lower()
        for key in self.SENSITIVE_KEYS:
            if key in lowered:
                # naive masking: replace after key= or key: patterns
                for sep in ('=', ':', '=>'):
                    needle = f"{key}{sep}"
                    if needle in lowered:
                        parts = msg.split(sep)
                        if len(parts) > 1:
                            parts[-1] = ' ' + self.MASK
                            record.msg = sep.join(parts)
                            return True
                record.msg = msg.replace(msg, self.MASK)
                return True
        return True

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup comprehensive logging configuration"""
    # Create log directory and standard sections
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        if DirectoryManager:
            DirectoryManager.create_logs_structure("logs")
    except Exception:
        # Non-fatal; logging will still proceed
        pass

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
            }
        },
        'filters': {
            'sensitive': {
                '()': SensitiveDataFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'filters': ['sensitive'],
                'stream': sys.stdout
            },
            # Root/system files
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/translation_system.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': f'{log_dir}/errors.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            # Section-specific files (always under logs/<section>)
            'file_training': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/training/training.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_data': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/data/data.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_monitoring': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/monitoring/monitoring.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_coordinator': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/coordinator/coordinator.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_decoder': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/decoder/decoder.log',
                'maxBytes': 10485760,
                'backupCount': 5
            },
            'file_vocabulary': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filters': ['sensitive'],
                'filename': 'logs/vocabulary/vocabulary.log',
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