# utils/rate_limiter.py
# IMPORTANT: This is a simple in-memory rate limiter suitable only for single-process deployments.
# In a distributed or multi-process environment (e.g., using gunicorn, Kubernetes),
# this will not work correctly. A centralized solution like Redis would be required.
from collections import defaultdict
import time
import threading
from typing import Dict, Tuple

class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        """Check if request is allowed"""
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < 3600  # Keep last hour
            ]
            
            # Check minute limit
            recent_minute = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < 60
            ]
            
            if len(recent_minute) >= self.requests_per_minute:
                return False, "Rate limit exceeded: too many requests per minute"
            
            # Check hour limit
            if len(self.requests[client_id]) >= self.requests_per_hour:
                return False, "Rate limit exceeded: too many requests per hour"
            
            # Allow request
            self.requests[client_id].append(current_time)
            return True, "OK"
    
    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """Get statistics for a client"""
        current_time = time.time()
        
        with self.lock:
            total = len(self.requests[client_id])
            last_minute = len([
                req for req in self.requests[client_id]
                if current_time - req < 60
            ])
            last_hour = len([
                req for req in self.requests[client_id]
                if current_time - req < 3600
            ])
        
        return {
            'total_requests': total,
            'last_minute': last_minute,
            'last_hour': last_hour
        }