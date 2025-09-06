# utils/rate_limiter.py
# Rate limiter with Redis support for distributed deployments
import time
import threading
import logging
import os
from collections import defaultdict
from typing import Dict, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter for API endpoints with Redis support for distributed deployments.
    Falls back to in-memory storage if Redis is not available.
    """
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000,
                 redis_url: Optional[str] = None):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
            requests_per_hour: Maximum requests allowed per hour
            redis_url: Redis URL for distributed rate limiting (e.g., redis://localhost:6379/0)
                       If None, falls back to in-memory storage
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.use_redis = False
        
        # Try to use Redis via RedisManager
        try:
            # Import Redis manager
            from utils.redis_manager import RedisManager
            
            # Get Redis manager instance
            self.redis_manager = RedisManager.get_instance()
            
            # Override default URL if provided
            if redis_url:
                self.redis_manager.default_url = redis_url
            
            # Test Redis connection
            client = self.redis_manager.get_client()
            if client:
                self.redis = client
                self.use_redis = True
                logger.info(f"Rate limiter using Redis via RedisManager")
            else:
                logger.warning("Redis connection not available, falling back to in-memory storage")
        except ImportError:
            logger.warning("Redis manager not available, falling back to in-memory storage")
        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
        
        # Fallback to in-memory storage if Redis is not available
        if not self.use_redis:
            logger.info("Rate limiter using in-memory storage")
            self.requests = defaultdict(list)
            self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Tuple of (is_allowed, message)
        """
        if self.use_redis and self.redis:
            return self._is_allowed_redis(client_id)
        else:
            return self._is_allowed_memory(client_id)
    
    def _is_allowed_redis(self, client_id: str) -> Tuple[bool, str]:
        """Redis-based rate limiting implementation"""
        current_time = int(time.time())
        minute_key = f"rate_limit:{client_id}:minute"
        hour_key = f"rate_limit:{client_id}:hour"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            
            # Add current timestamp to sorted sets
            pipe.zadd(minute_key, {str(current_time): current_time})
            pipe.zadd(hour_key, {str(current_time): current_time})
            
            # Remove timestamps older than the time window
            pipe.zremrangebyscore(minute_key, 0, current_time - 60)
            pipe.zremrangebyscore(hour_key, 0, current_time - 3600)
            
            # Get counts
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            
            # Set expiration to prevent memory leaks
            pipe.expire(minute_key, 120)  # 2 minutes
            pipe.expire(hour_key, 4000)   # ~1 hour
            
            # Execute pipeline
            results = pipe.execute()
            minute_count = results[4]
            hour_count = results[5]
            
            # Check limits
            if minute_count > self.requests_per_minute:
                return False, "Rate limit exceeded: too many requests per minute"
            
            if hour_count > self.requests_per_hour:
                return False, "Rate limit exceeded: too many requests per hour"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to in-memory if Redis fails
            return self._is_allowed_memory(client_id)
    
    def _is_allowed_memory(self, client_id: str) -> Tuple[bool, str]:
        """In-memory rate limiting implementation"""
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
        """
        Get statistics for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Dictionary with request statistics
        """
        if self.use_redis and self.redis:
            return self._get_client_stats_redis(client_id)
        else:
            return self._get_client_stats_memory(client_id)
    
    def _get_client_stats_redis(self, client_id: str) -> Dict[str, int]:
        """Redis-based client statistics"""
        current_time = int(time.time())
        minute_key = f"rate_limit:{client_id}:minute"
        hour_key = f"rate_limit:{client_id}:hour"
        
        try:
            # Use Redis pipeline for better performance
            pipe = self.redis.pipeline()
            pipe.zcard(minute_key)
            pipe.zcard(hour_key)
            results = pipe.execute()
            
            last_minute = results[0]
            last_hour = results[1]
            
            return {
                'total_requests': last_hour,  # Approximation
                'last_minute': last_minute,
                'last_hour': last_hour
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            # Fall back to in-memory if Redis fails
            return self._get_client_stats_memory(client_id)
    
    def _get_client_stats_memory(self, client_id: str) -> Dict[str, int]:
        """In-memory client statistics"""
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