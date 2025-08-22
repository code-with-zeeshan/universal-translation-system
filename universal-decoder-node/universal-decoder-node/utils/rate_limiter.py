# universal-decoder-node/universal_decoder_node/utils/rate_limiter.py
import time
import logging
import threading
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
        
        # In-memory storage for rate limiting
        self._minute_counters: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._hour_counters: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.RLock()
        
        # Try to use Redis if available
        try:
            # Try to import the main project's RedisManager
            try:
                from utils.redis_manager import RedisManager
                self.redis_manager = RedisManager.get_instance()
                
                # Override default URL if provided
                if redis_url:
                    self.redis_manager.default_url = redis_url
                
                # Test Redis connection
                if self.redis_manager.get_client():
                    self.use_redis = True
                    logger.info(f"Rate limiter using Redis via RedisManager")
            except ImportError:
                # If main project's RedisManager is not available, try to use redis directly
                import redis
                self.redis_url = redis_url or os.environ.get("REDIS_URL")
                if self.redis_url:
                    self.redis = redis.from_url(
                        self.redis_url,
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        retry_on_timeout=True
                    )
                    self.redis.ping()  # Test connection
                    self.use_redis = True
                    logger.info(f"Rate limiter using Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting, using in-memory storage: {e}")
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a client is allowed to make a request.
        
        Args:
            client_id: Client identifier (e.g., IP address, API key)
            
        Returns:
            True if the client is allowed, False otherwise
        """
        if self.use_redis:
            try:
                return self._is_allowed_redis(client_id)
            except Exception as e:
                logger.error(f"Redis rate limiting failed, falling back to in-memory: {e}")
                return self._is_allowed_memory(client_id)
        else:
            return self._is_allowed_memory(client_id)
    
    def _is_allowed_memory(self, client_id: str) -> bool:
        """In-memory rate limiting implementation"""
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        with self._lock:
            # Clean up old counters
            self._cleanup_counters()
            
            # Check minute limit
            minute_count = self._minute_counters[client_id][current_minute]
            if minute_count >= self.requests_per_minute:
                return False
            
            # Check hour limit
            hour_count = self._hour_counters[client_id][current_hour]
            if hour_count >= self.requests_per_hour:
                return False
            
            # Increment counters
            self._minute_counters[client_id][current_minute] += 1
            self._hour_counters[client_id][current_hour] += 1
            
            return True
    
    def _is_allowed_redis(self, client_id: str) -> bool:
        """Redis-based rate limiting implementation"""
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        # Use RedisManager if available
        if hasattr(self, 'redis_manager'):
            redis_client = self.redis_manager.get_client()
            if not redis_client:
                return self._is_allowed_memory(client_id)
                
            pipeline = redis_client.pipeline()
            
            # Minute key with TTL of 2 minutes
            minute_key = f"rate_limit:{client_id}:minute:{current_minute}"
            pipeline.incr(minute_key)
            pipeline.expire(minute_key, 120)
            
            # Hour key with TTL of 2 hours
            hour_key = f"rate_limit:{client_id}:hour:{current_hour}"
            pipeline.incr(hour_key)
            pipeline.expire(hour_key, 7200)
            
            # Execute pipeline
            results = pipeline.execute()
            minute_count, _, hour_count, _ = results
            
            return minute_count <= self.requests_per_minute and hour_count <= self.requests_per_hour
        else:
            # Use direct Redis connection
            pipeline = self.redis.pipeline()
            
            # Minute key with TTL of 2 minutes
            minute_key = f"rate_limit:{client_id}:minute:{current_minute}"
            pipeline.incr(minute_key)
            pipeline.expire(minute_key, 120)
            
            # Hour key with TTL of 2 hours
            hour_key = f"rate_limit:{client_id}:hour:{current_hour}"
            pipeline.incr(hour_key)
            pipeline.expire(hour_key, 7200)
            
            # Execute pipeline
            results = pipeline.execute()
            minute_count, _, hour_count, _ = results
            
            return minute_count <= self.requests_per_minute and hour_count <= self.requests_per_hour
    
    def _cleanup_counters(self):
        """Clean up old counters to prevent memory leaks"""
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        # Keep only the last 5 minutes
        for client_id in list(self._minute_counters.keys()):
            self._minute_counters[client_id] = {
                minute: count
                for minute, count in self._minute_counters[client_id].items()
                if minute >= current_minute - 5
            }
            
            # Remove client if no counters left
            if not self._minute_counters[client_id]:
                del self._minute_counters[client_id]
        
        # Keep only the last 3 hours
        for client_id in list(self._hour_counters.keys()):
            self._hour_counters[client_id] = {
                hour: count
                for hour, count in self._hour_counters[client_id].items()
                if hour >= current_hour - 3
            }
            
            # Remove client if no counters left
            if not self._hour_counters[client_id]:
                del self._hour_counters[client_id]
    
    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """
        Get statistics for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with minute_count and hour_count
        """
        if self.use_redis:
            try:
                return self._get_client_stats_redis(client_id)
            except Exception as e:
                logger.error(f"Redis stats retrieval failed, falling back to in-memory: {e}")
                return self._get_client_stats_memory(client_id)
        else:
            return self._get_client_stats_memory(client_id)
    
    def _get_client_stats_memory(self, client_id: str) -> Dict[str, int]:
        """Get client statistics from in-memory storage"""
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        with self._lock:
            minute_count = self._minute_counters[client_id][current_minute]
            hour_count = self._hour_counters[client_id][current_hour]
            
            return {
                "minute_count": minute_count,
                "hour_count": hour_count,
                "minute_limit": self.requests_per_minute,
                "hour_limit": self.requests_per_hour
            }
    
    def _get_client_stats_redis(self, client_id: str) -> Dict[str, int]:
        """Get client statistics from Redis"""
        current_minute = int(time.time() / 60)
        current_hour = int(time.time() / 3600)
        
        # Use RedisManager if available
        if hasattr(self, 'redis_manager'):
            redis_client = self.redis_manager.get_client()
            if not redis_client:
                return self._get_client_stats_memory(client_id)
                
            minute_key = f"rate_limit:{client_id}:minute:{current_minute}"
            hour_key = f"rate_limit:{client_id}:hour:{current_hour}"
            
            minute_count = int(redis_client.get(minute_key) or 0)
            hour_count = int(redis_client.get(hour_key) or 0)
        else:
            # Use direct Redis connection
            minute_key = f"rate_limit:{client_id}:minute:{current_minute}"
            hour_key = f"rate_limit:{client_id}:hour:{current_hour}"
            
            minute_count = int(self.redis.get(minute_key) or 0)
            hour_count = int(self.redis.get(hour_key) or 0)
        
        return {
            "minute_count": minute_count,
            "hour_count": hour_count,
            "minute_limit": self.requests_per_minute,
            "hour_limit": self.requests_per_hour
        }