# utils/lazy_loader.py
"""
Lazy loading utilities for the Universal Translation System.
This module provides tools for lazy initialization of resource-intensive components.
"""

import functools
import inspect
import logging
import threading
from typing import Any, Callable, Dict, Optional, TypeVar, cast, Type

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyObject:
    """
    A wrapper for lazily initializing an object.
    The object is only created when first accessed.
    """
    
    def __init__(self, factory: Callable[[], T]):
        """
        Initialize lazy object.
        
        Args:
            factory: Function that creates the object
        """
        self.__dict__['_factory'] = factory
        self.__dict__['_object'] = None
        self.__dict__['_lock'] = threading.RLock()
        
    def _initialize(self) -> T:
        """Initialize the wrapped object if not already initialized."""
        if self.__dict__['_object'] is None:
            with self.__dict__['_lock']:
                if self.__dict__['_object'] is None:
                    try:
                        self.__dict__['_object'] = self.__dict__['_factory']()
                    except Exception as e:
                        logger.error(f"Error initializing lazy object: {e}")
                        raise
        return self.__dict__['_object']
        
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped object."""
        return getattr(self._initialize(), name)
        
    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignment to the wrapped object."""
        setattr(self._initialize(), name, value)
        
    def __delattr__(self, name: str) -> None:
        """Forward attribute deletion to the wrapped object."""
        delattr(self._initialize(), name)
        
    def __call__(self, *args, **kwargs) -> Any:
        """Forward calls to the wrapped object."""
        return self._initialize()(*args, **kwargs)
        
    def __repr__(self) -> str:
        """Return string representation."""
        obj = self.__dict__['_object']
        return f"<LazyObject: {'uninitialized' if obj is None else repr(obj)}>"

    def __enter__(self):
        return self._initialize()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def reset(self) -> None:
        """Reset the lazy object to uninitialized state."""
        with self.__dict__['_lock']:
            self.__dict__['_object'] = None




def lazy_property(func: Callable[[Any], T]) -> property:
    """
    Decorator for creating lazy properties.
    The property value is computed only when first accessed.
    
    Args:
        func: Function that computes the property value
        
    Returns:
        Property descriptor
    """
    attr_name = f"_lazy_{func.__name__}"
    
    @functools.wraps(func)
    def lazy_wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
        
    return property(lazy_wrapper)

# Add a max_size parameter to limit cache growth
def lazy_function(func: Callable[..., T], max_size: int = 0) -> Callable[..., T]:
    """
    Decorator for creating lazy functions.
    The function result is cached after the first call with the same arguments.
    
    Args:
        func: Function to decorate
        max_size: Optional max cache size (0 = unlimited)
        
    Returns:
        Decorated function
    """
    cache: Dict[tuple, Any] = {}
    lock = threading.RLock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from the arguments
        key_args = args
        key_kwargs = tuple(sorted(kwargs.items()))
        cache_key = (key_args, key_kwargs)
        
        # Check if result is already cached
        with lock:
            if cache_key in cache:
                return cache[cache_key]
        
        # Compute result outside lock
        result = func(*args, **kwargs)
        
        # Store result with size management
        with lock:
            if max_size > 0 and len(cache) >= max_size:
                # Remove oldest entry (first item in dict)
                if cache:
                    cache.pop(next(iter(cache)))
            cache[cache_key] = result
        return result

    # Add a method to clear the cache
    def clear_cache():
        with lock:
            cache.clear()
            
    wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
    
    return wrapper

class LazyClass:
    """
    Base class for creating lazy-loaded classes.
    Subclasses can define _lazy_init() to perform lazy initialization.
    """
    
    def __init__(self):
        self._initialized = False
        self._init_lock = threading.RLock()
        
    def _ensure_initialized(self):
        """Ensure the object is initialized."""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    try:
                        self._lazy_init()
                        self._initialized = True
                    except Exception as e:
                        logger.error(f"Error in lazy initialization: {e}")
                        raise
                        
    def _lazy_init(self):
        """
        Perform lazy initialization.
        Subclasses should override this method.
        """
        pass


def lazy_import(module_name: str) -> Any:
    """
    Lazily import a module.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Lazy module proxy
    """
    return LazyObject(lambda: __import__(module_name, fromlist=['']))


def lazy_singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator for creating lazy singletons.
    The singleton instance is created only when first accessed.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
    """
    instance_lock = threading.RLock()
    instance = None
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is None:
            with instance_lock:
                if instance is None:
                    instance = cls(*args, **kwargs)
        return instance
        
    return cast(Type[T], wrapper)
