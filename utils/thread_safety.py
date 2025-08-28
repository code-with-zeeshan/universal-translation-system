# utils/thread_safety.py
"""
Thread safety utilities for the Universal Translation System.
This module provides tools for documenting and enforcing thread safety.
"""

import functools
import threading
import inspect
from typing import Callable, Any, Dict, Optional, Type, TypeVar

T = TypeVar('T')

# Thread safety levels
THREAD_SAFETY_NONE = "none"  # Not thread-safe
THREAD_SAFETY_EXTERNAL = "external"  # Thread-safe with external synchronization
THREAD_SAFETY_INTERNAL = "internal"  # Thread-safe with internal synchronization
THREAD_SAFETY_IMMUTABLE = "immutable"  # Thread-safe because immutable

# Registry of thread safety documentation
_thread_safety_registry: Dict[Type, Dict[str, str]] = {}


def document_thread_safety(cls: Type[T], level: str, description: str = "") -> Type[T]:
    """
    Document thread safety for a class.
    
    Args:
        cls: Class to document
        level: Thread safety level
        description: Additional description
        
    Returns:
        The class (for chaining)
    """
    if cls not in _thread_safety_registry:
        _thread_safety_registry[cls] = {}
        
    _thread_safety_registry[cls]["class"] = level
    _thread_safety_registry[cls]["description"] = description
    
    # Add thread safety information to class docstring
    if cls.__doc__:
        cls.__doc__ = f"{cls.__doc__}\n\nThread Safety: {level}\n{description}"
    else:
        cls.__doc__ = f"Thread Safety: {level}\n{description}"
        
    return cls


def document_method_thread_safety(method: Callable, level: str, description: str = "") -> Callable:
    """
    Document thread safety for a method.
    
    Args:
        method: Method to document
        level: Thread safety level
        description: Additional description
        
    Returns:
        The method (for chaining)
    """
    cls = method.__qualname__.split('.')[0]
    method_name = method.__name__
    
    if cls not in _thread_safety_registry:
        _thread_safety_registry[cls] = {}
        
    _thread_safety_registry[cls][method_name] = level
    
    # Add thread safety information to method docstring
    if method.__doc__:
        method.__doc__ = f"{method.__doc__}\n\nThread Safety: {level}\n{description}"
    else:
        method.__doc__ = f"Thread Safety: {level}\n{description}"
        
    return method


def thread_safe(method: Callable) -> Callable:
    """
    Decorator to make a method thread-safe using a lock.
    
    Args:
        method: Method to make thread-safe
        
    Returns:
        Thread-safe method
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Get or create lock
        if not hasattr(self, '_thread_safe_lock'):
            self._thread_safe_lock = threading.RLock()
            
        # Execute with lock
        with self._thread_safe_lock:
            return method(self, *args, **kwargs)
            
    # Document thread safety
    document_method_thread_safety(wrapper, THREAD_SAFETY_INTERNAL, 
                                "This method is thread-safe with internal synchronization.")
    
    return wrapper


def get_thread_safety_info(cls: Type) -> Dict[str, str]:
    """
    Get thread safety information for a class.
    
    Args:
        cls: Class to get information for
        
    Returns:
        Dictionary of thread safety information
    """
    return _thread_safety_registry.get(cls, {})


def generate_thread_safety_report() -> Dict[str, Dict[str, str]]:
    """
    Generate a report of thread safety for all documented classes.
    
    Returns:
        Dictionary mapping class names to thread safety information
    """
    report = {}
    
    for cls, info in _thread_safety_registry.items():
        class_name = cls.__name__
        report[class_name] = info.copy()
        
    return report
