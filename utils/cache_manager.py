# utils/cache_manager.py
"""
Advanced caching utilities for the Universal Translation System.
This module provides tools for caching data with various eviction policies.
"""

import time
import threading
import logging
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, List, Tuple, Union, Generic
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    
    def __init__(self, value: V, ttl: Optional[float] = None):
        """
        Initialize cache entry.
        
        Args:
            value: The cached value
            ttl: Time to live in seconds (None for no expiration)
        """
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        
    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl


class Cache(Generic[K, V]):
    """
    Advanced cache with multiple eviction policies.
    Thread-safe and supports time-based expiration.
    """
    
    def __init__(
        self,
        max_size: Optional[int] = 1000,
        ttl: Optional[float] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries (None for unlimited)
            ttl: Default time to live in seconds (None for no expiration)
            eviction_policy: Policy for evicting entries when the cache is full
            on_evict: Callback function called when an entry is evicted
        """
        self._cache: Dict[K, CacheEntry[V]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._eviction_policy = eviction_policy
        self._on_evict = on_evict
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.debug(
            f"Initialized cache with max_size={max_size}, "
            f"ttl={ttl}, policy={eviction_policy.value}"
        )
        
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._evict(key)
                    self._misses += 1
                    return default
                
                # Update access metadata
                entry.access()
                
                # Move to end for LRU
                if self._eviction_policy == EvictionPolicy.LRU:
                    self._cache.move_to_end(key)
                
                self._hits += 1
                return entry.value
            
            self._misses += 1
            return default
            
    def set(
        self, 
        key: K, 
        value: V, 
        ttl: Optional[float] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default)
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            
            # Use default TTL if not specified
            if ttl is None:
                ttl = self._ttl
                
            # Create entry
            entry = CacheEntry(value, ttl)
            
            # Check if key already exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._cache[key] = entry
                
                # Move to end for LRU/FIFO
                if self._eviction_policy in (EvictionPolicy.LRU, EvictionPolicy.FIFO):
                    self._cache.move_to_end(key)
                    
                return
                
            # Check if cache is full
            if self._max_size is not None and len(self._cache) >= self._max_size:
                self._evict_one()
                
            # Add new entry
            self._cache[key] = entry
            
            # Move to end for LRU/FIFO
            if self._eviction_policy in (EvictionPolicy.LRU, EvictionPolicy.FIFO):
                self._cache.move_to_end(key)
                
    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                
                # Call eviction callback
                if self._on_evict:
                    try:
                        self._on_evict(key, entry.value)
                    except Exception as e:
                        logger.error(f"Error in eviction callback: {e}")
                        
                return True
                
            return False
            
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            # Call eviction callback for each entry
            if self._on_evict:
                for key, entry in self._cache.items():
                    try:
                        self._on_evict(key, entry.value)
                    except Exception as e:
                        logger.error(f"Error in eviction callback: {e}")
                        
            self._cache.clear()
            logger.debug("Cache cleared")
            
    def keys(self) -> List[K]:
        """Get all keys in the cache."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            return list(self._cache.keys())
            
    def values(self) -> List[V]:
        """Get all values in the cache."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            return [entry.value for entry in self._cache.values()]
            
    def items(self) -> List[Tuple[K, V]]:
        """Get all items in the cache."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            return [(key, entry.value) for key, entry in self._cache.items()]
            
    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            return len(self._cache)
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "eviction_policy": self._eviction_policy.value,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "evictions": self._evictions
            }
            
    def _clean_expired(self) -> None:
        """Clean expired entries."""
        if self._ttl is None:
            return
        
        # Only check a subset of entries each time to avoid performance issues
        # with large caches
        now = time.time()
        sample_size = min(len(self._cache), 20)  # Check at most 20 entries
    
        # Sample random keys
        import random
        keys_to_check = random.sample(list(self._cache.keys()), sample_size)
    
        expired_keys = [
            key for key in keys_to_check 
            if self._cache[key].is_expired()
        ]
    
        for key in expired_keys:
            self._evict(key)

            
    def _evict_one(self) -> None:
        """Evict one entry based on the eviction policy."""
        if not self._cache:
            return
            
        key_to_evict = None
        
        if self._eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used (first item in OrderedDict)
            key_to_evict = next(iter(self._cache))
        elif self._eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self._cache.items(),
                key=lambda item: item[1].access_count
            )[0]
        elif self._eviction_policy == EvictionPolicy.FIFO:
            # Evict first in (first item in OrderedDict)
            key_to_evict = next(iter(self._cache))
        elif self._eviction_policy == EvictionPolicy.TTL:
            # Evict oldest entry
            key_to_evict = min(
                self._cache.items(),
                key=lambda item: item[1].created_at
            )[0]
            
        if key_to_evict:
            self._evict(key_to_evict)
            
    def _evict(self, key: K) -> None:
        """Evict an entry."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._evictions += 1
            
            # Call eviction callback
            if self._on_evict:
                try:
                    self._on_evict(key, entry.value)
                except Exception as e:
                    logger.error(f"Error in eviction callback: {e}")


def cached(
    ttl: Optional[float] = None,
    max_size: Optional[int] = 1000,
    key_fn: Optional[Callable[..., Any]] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_fn: Function to generate cache keys (defaults to args and kwargs)
        
    Returns:
        Decorated function
    """
    cache = Cache(max_size=max_size, ttl=ttl)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default key is based on function name, args, and kwargs
                key_args = args
                key_kwargs = tuple(sorted(kwargs.items()))
                cache_key = (func.__name__, key_args, key_kwargs)
                
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
                
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result)
            
            return result
            
        # Add cache management methods
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        
        return wrapper
        
    return decorator

def get_hit_ratio(self) -> float:
    """Get the cache hit ratio."""
    total = self._hits + self._misses
    return self._hits / total if total > 0 else 0.0

def serialize(self) -> Dict[str, Any]:
    """Serialize cache to a dictionary."""
    with self._lock:
        return {
            "items": {
                str(key): {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "ttl": entry.ttl
                }
                for key, entry in self._cache.items()
                if not entry.is_expired()
            },
            "stats": self.stats()
        }
        
@classmethod
def deserialize(cls, data: Dict[str, Any], **kwargs) -> 'Cache':
    """Create a cache from serialized data."""
    cache = cls(**kwargs)
    
    for key_str, item_data in data.get("items", {}).items():
        entry = CacheEntry(
            item_data["value"],
            item_data.get("ttl")
        )
        entry.created_at = item_data.get("created_at", time.time())
        entry.last_accessed = item_data.get("last_accessed", time.time())
        entry.access_count = item_data.get("access_count", 0)
        
        # Try to convert key back to original type
        try:
            # Try as int, float, then fallback to string
            key = int(key_str)
        except ValueError:
            try:
                key = float(key_str)
            except ValueError:
                key = key_str
                
        cache._cache[key] = entry
        
    return cache

class CacheManager:
    """
    Global cache manager for the Universal Translation System.
    Manages multiple named caches.
    """
    
    def __init__(self):
        self._caches: Dict[str, Cache] = {}
        self._lock = threading.RLock()
        
    def get_cache(
        self,
        name: str,
        max_size: Optional[int] = 1000,
        ttl: Optional[float] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        on_evict: Optional[Callable[[Any, Any], None]] = None
    ) -> Cache:
        """
        Get or create a cache.
        
        Args:
            name: Cache name
            max_size: Maximum cache size
            ttl: Default time to live in seconds
            eviction_policy: Eviction policy
            on_evict: Callback function called when an entry is evicted
            
        Returns:
            Cache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = Cache(
                    max_size=max_size,
                    ttl=ttl,
                    eviction_policy=eviction_policy,
                    on_evict=on_evict
                )
            return self._caches[name]
            
    def clear_cache(self, name: str) -> bool:
        """
        Clear a cache.
        
        Args:
            name: Cache name
            
        Returns:
            True if the cache was found and cleared
        """
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                return True
            return False
            
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
                
    def remove_cache(self, name: str) -> bool:
        """
        Remove a cache.
        
        Args:
            name: Cache name
            
        Returns:
            True if the cache was found and removed
        """
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                del self._caches[name]
                return True
            return False
            
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary mapping cache names to statistics
        """
        with self._lock:
            return {name: cache.stats() for name, cache in self._caches.items()}


# Create a global cache manager instance
cache_manager = CacheManager()
