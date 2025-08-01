# utils/logging_config.py
import logging
import logging.config
from pathlib import Path
import sys

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup comprehensive logging configuration"""
    
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
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
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': f'{log_dir}/translation_system.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': f'{log_dir}/errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'data': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'vocabulary': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
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