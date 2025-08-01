# utils/shutdown_handler.py
import signal
import sys
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class GracefulShutdown:
    def __init__(self, cleanup_func: Optional[Callable] = None):
        self.shutdown_event = threading.Event()
        self.cleanup_func = cleanup_func
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        
        if self.cleanup_func:
            try:
                logger.info("Running cleanup function...")
                self.cleanup_func()
                logger.info("Cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        # Restore original handlers
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        
        # Exit gracefully
        sys.exit(0)
    
    def should_stop(self) -> bool:
        """Check if shutdown was requested"""
        return self.shutdown_event.is_set()
    
    def wait_for_shutdown(self):
        """Block until shutdown is requested"""
        self.shutdown_event.wait()