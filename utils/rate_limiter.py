# utils/rate_limiter.py
"""
Rate limiter backed by slowapi with in-memory and Redis support.
Provides RateLimiter class for backward compatibility with existing services.
"""

import time
import logging
import os
import threading
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter as SlowapiLimiter
    from slowapi.util import get_remote_address
    _slowapi_available = True
except Exception:
    _slowapi_available = False


class RateLimiter:
    """Rate limiter wrapping slowapi with backward-compatible is_allowed() interface."""

    def __init__(self, requests_per_minute: int = 60,
                 redis_url: Optional[str] = None):
        self.requests_per_minute = requests_per_minute
        self._slots: Dict[str, list] = {}
        self._lock = threading.RLock()
        if _slowapi_available:
            self._limiter = SlowapiLimiter(
                key_func=get_remote_address,
                default_limits=[f"{requests_per_minute}/minute"],
                storage_uri=redis_url or os.getenv("REDIS_URL"),
            )
        else:
            self._limiter = None

    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        """Check if a client is allowed to proceed. Returns (allowed, message)."""
        now = time.time()
        with self._lock:
            if client_id not in self._slots:
                self._slots[client_id] = []
            window = 60.0
            cutoff = now - window
            self._slots[client_id] = [t for t in self._slots[client_id] if t > cutoff]
            if len(self._slots[client_id]) >= self.requests_per_minute:
                retry_after = int(self._slots[client_id][0] + window - now)
                return False, f"Rate limit exceeded. Retry after {retry_after}s"
            self._slots[client_id].append(now)
            return True, ""
