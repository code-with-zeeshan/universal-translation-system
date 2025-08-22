# utils/redis_manager.py
import os
import time
import logging
import threading
import json
import msgpack
from typing import Any, Dict, List, Optional, Union, Callable
import redis
from redis.connection import ConnectionPool

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Centralized Redis connection manager with connection pooling, health checks,
    and standardized serialization.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Health monitoring
    - Configurable key prefixes
    - MessagePack serialization
    - Timeout configuration
    """
    
    _instance = None
    _pools: Dict[str, ConnectionPool] = {}
    _clients: Dict[str, redis.Redis] = {}
    _health_check_thread = None
    _stop_health_check = threading.Event()
    
    @classmethod
    def get_instance(cls) -> 'RedisManager':
        """Get singleton instance of RedisManager"""
        if cls._instance is None:
            cls._instance = RedisManager()
        return cls._instance
    
    def __init__(self):
        self.default_url = os.environ.get("REDIS_URL")
        self.key_prefix = os.environ.get("REDIS_KEY_PREFIX", "translation:")
        self.connection_timeout = int(os.environ.get("REDIS_CONN_TIMEOUT", "2"))
        self.read_timeout = int(os.environ.get("REDIS_READ_TIMEOUT", "2"))
        self.health_check_interval = int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        self.use_msgpack = os.environ.get("REDIS_USE_MSGPACK", "true").lower() == "true"
        
        # Start health check thread if enabled
        if os.environ.get("REDIS_HEALTH_CHECK_ENABLED", "true").lower() == "true":
            self._start_health_check()
    
    def get_client(self, url: Optional[str] = None) -> Optional[redis.Redis]:
        """
        Get a Redis client from the connection pool.
        
        Args:
            url: Redis URL. If None, uses the default URL from environment.
            
        Returns:
            Redis client or None if connection fails
        """
        url = url or self.default_url
        
        if not url:
            logger.warning("No Redis URL provided and no default URL set")
            return None
        
        # Return existing client if available
        if url in self._clients:
            return self._clients[url]
        
        # Create new connection pool and client
        try:
            if url not in self._pools:
                self._pools[url] = redis.ConnectionPool.from_url(
                    url,
                    socket_connect_timeout=self.connection_timeout,
                    socket_timeout=self.read_timeout,
                    retry_on_timeout=True
                )
            
            client = redis.Redis(connection_pool=self._pools[url])
            # Test connection
            client.ping()
            self._clients[url] = client
            logger.info(f"Redis connection established to {url}")
            return client
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis at {url}: {e}")
            return None
    
    def _start_health_check(self):
        """Start background thread for Redis health checks"""
        if self._health_check_thread is not None and self._health_check_thread.is_alive():
            return
            
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="RedisHealthCheck"
        )
        self._health_check_thread.start()
        logger.info("Redis health check thread started")
    
    def _health_check_loop(self):
        """Background thread that periodically checks Redis health"""
        while not self._stop_health_check.is_set():
            for url, client in list(self._clients.items()):
                try:
                    client.ping()
                except redis.RedisError as e:
                    logger.warning(f"Redis health check failed for {url}: {e}")
                    # Remove client from cache to force reconnection
                    self._clients.pop(url, None)
            
            # Sleep until next check
            self._stop_health_check.wait(self.health_check_interval)
    
    def stop_health_check(self):
        """Stop the health check thread"""
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._stop_health_check.set()
            self._health_check_thread.join(timeout=1.0)
            logger.info("Redis health check thread stopped")
    
    def get_prefixed_key(self, key: str) -> str:
        """Get key with configured prefix"""
        return f"{self.key_prefix}{key}"
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        Set a value in Redis with serialization.
        
        Args:
            key: Redis key (will be prefixed)
            value: Value to store
            ex: Expiration time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        client = self.get_client()
        if not client:
            return False
            
        prefixed_key = self.get_prefixed_key(key)
        
        try:
            # Serialize based on configuration
            if self.use_msgpack:
                serialized = msgpack.packb(value, use_bin_type=True)
            else:
                serialized = json.dumps(value)
                
            return client.set(prefixed_key, serialized, ex=ex)
        except Exception as e:
            logger.error(f"Error setting Redis key {prefixed_key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis with deserialization.
        
        Args:
            key: Redis key (will be prefixed)
            default: Default value if key doesn't exist
            
        Returns:
            Deserialized value or default
        """
        client = self.get_client()
        if not client:
            return default
            
        prefixed_key = self.get_prefixed_key(key)
        
        try:
            result = client.get(prefixed_key)
            if result is None:
                return default
                
            # Deserialize based on configuration
            if self.use_msgpack:
                return msgpack.unpackb(result, raw=False)
            else:
                return json.loads(result)
        except Exception as e:
            logger.error(f"Error getting Redis key {prefixed_key}: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a key from Redis"""
        client = self.get_client()
        if not client:
            return False
            
        prefixed_key = self.get_prefixed_key(key)
        
        try:
            return client.delete(prefixed_key) > 0
        except Exception as e:
            logger.error(f"Error deleting Redis key {prefixed_key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis"""
        client = self.get_client()
        if not client:
            return False
            
        prefixed_key = self.get_prefixed_key(key)
        
        try:
            return client.exists(prefixed_key) > 0
        except Exception as e:
            logger.error(f"Error checking Redis key {prefixed_key}: {e}")
            return False
    
    def pipeline(self) -> Optional[redis.client.Pipeline]:
        """Get a Redis pipeline for batch operations"""
        client = self.get_client()
        if not client:
            return None
            
        return client.pipeline()
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a Redis function with automatic retry on connection errors.
        
        Args:
            func: Redis function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function or None if all retries fail
        """
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis operation failed, retrying ({attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Redis operation failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                return None