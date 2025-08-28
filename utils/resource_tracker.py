# utils/resource_tracker.py
"""
Resource tracking utilities for the Universal Translation System.
This module provides tools for tracking and managing resource usage.
"""

import gc
import logging
import threading
import time
import weakref
from typing import Dict, Set, Any, Optional, List, Tuple, Callable
import os
import psutil
import tracemalloc

logger = logging.getLogger(__name__)


class ResourceTracker:
    """
    Tracks resource usage and helps identify memory leaks.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize resource tracker.
        
        Args:
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled
        self._tracked_objects: Dict[int, Tuple[weakref.ref, str, int]] = {}
        self._lock = threading.RLock()
        self._tracemalloc_started = False
        
        # Start tracemalloc if enabled
        if self.enabled:
            self.start_tracemalloc()
            
    def track(self, obj: Any, description: str = "") -> None:
        """
        Track an object to detect leaks.
        
        Args:
            obj: Object to track
            description: Description of the object
        """
        if not self.enabled:
            return
            
        with self._lock:
            obj_id = id(obj)
            self._tracked_objects[obj_id] = (
                weakref.ref(obj, lambda ref: self._object_finalized(obj_id)),
                description or type(obj).__name__,
                len(gc.get_referrers(obj))
            )
            
    def untrack(self, obj: Any) -> None:
        """
        Stop tracking an object.
        
        Args:
            obj: Object to stop tracking
        """
        if not self.enabled:
            return
            
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._tracked_objects:
                del self._tracked_objects[obj_id]
                
    def _object_finalized(self, obj_id: int) -> None:
        """Called when a tracked object is garbage collected."""
        with self._lock:
            if obj_id in self._tracked_objects:
                del self._tracked_objects[obj_id]
                
    def get_tracked_objects(self) -> List[Tuple[Any, str, int]]:
        """
        Get all tracked objects.
        
        Returns:
            List of (object, description, refcount) tuples
        """
        if not self.enabled:
            return []
            
        with self._lock:
            result = []
            for obj_id, (ref, description, initial_refs) in list(self._tracked_objects.items()):
                obj = ref()
                if obj is not None:
                    current_refs = len(gc.get_referrers(obj))
                    result.append((obj, description, current_refs))
            return result
            
    def get_leaks(self) -> List[Tuple[Any, str, int, int]]:
        """
        Get potential memory leaks (objects with increasing reference counts).
        
        Returns:
            List of (object, description, initial_refs, current_refs) tuples
        """
        if not self.enabled:
            return []
            
        with self._lock:
            result = []
            for obj_id, (ref, description, initial_refs) in list(self._tracked_objects.items()):
                obj = ref()
                if obj is not None:
                    current_refs = len(gc.get_referrers(obj))
                    if current_refs > initial_refs + 2:  # Allow some overhead
                        result.append((obj, description, initial_refs, current_refs))
            return result
            
    def print_leaks(self) -> None:
        """Print potential memory leaks."""
        if not self.enabled:
            return
            
        leaks = self.get_leaks()
        if leaks:
            logger.warning(f"Potential memory leaks detected: {len(leaks)}")
            for obj, description, initial_refs, current_refs in leaks:
                logger.warning(
                    f"  {description}: {initial_refs} -> {current_refs} refs"
                )
        else:
            logger.info("No memory leaks detected")
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms": memory_info.vms,
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "tracked_objects": len(self._tracked_objects) if self.enabled else 0
        }
        
    def start_tracemalloc(self) -> None:
        """Start tracemalloc for detailed memory tracking."""
        if not self.enabled:
            return
            
        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            logger.info("Started tracemalloc")
            
    def stop_tracemalloc(self) -> None:
        """Stop tracemalloc."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            logger.info("Stopped tracemalloc")
            
    def get_tracemalloc_snapshot(self) -> Optional[List[Tuple[str, int]]]:
        """
        Get a snapshot of memory allocations.
        
        Returns:
            List of (trace, size) tuples or None if tracemalloc is not started
        """
        if not self._tracemalloc_started:
            return None
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return [(str(stat.traceback), stat.size) for stat in top_stats[:10]]
        
    def print_tracemalloc_snapshot(self) -> None:
        """Print the top memory allocations."""
        if not self._tracemalloc_started:
            logger.warning("Tracemalloc not started")
            return
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.info("Top 10 memory allocations:")
        for stat in top_stats[:10]:
            logger.info(f"{stat.size / 1024:.1f} KB: {stat.traceback}")
            
    def start_periodic_leak_check(self, interval: int = 3600) -> None:
        """Start periodic leak checking."""
        if not self.enabled:
            return
        
        def check_leaks():
            while True:
                time.sleep(interval)
                self.print_leaks()
            
        thread = threading.Thread(target=check_leaks, daemon=True)
        thread.start()
        self._leak_check_thread = thread

    def set_memory_alert(self, threshold_mb: int, 
                        callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set an alert for when memory usage exceeds a threshold."""
        if not self.enabled:
            return
        
        def check_memory():
            while True:
                usage = self.get_memory_usage()
                if usage["rss_mb"] > threshold_mb:
                    callback(usage)
                time.sleep(60)  # Check every minute
            
        thread = threading.Thread(target=check_memory, daemon=True)
        thread.start()
        self._memory_alert_thread = thread

    def generate_object_lifecycle_report(self) -> Dict[str, Any]:
        """Generate a report of object lifecycles."""
        if not self.enabled:
            return {}
        
        with self._lock:
            return {
                "tracked_objects": len(self._tracked_objects),
                "object_types": {
                   type_name: count
                   for type_name, count in self._count_by_type().items()
                },
                "oldest_objects": [
                    {
                        "type": type(obj).__name__,
                        "description": description,
                        "age": time.time() - self._tracked_objects[id(obj)][3],
                        "refs": len(gc.get_referrers(obj))
                    }
                    for obj, description, _ in sorted(
                        self.get_tracked_objects(),
                        key=lambda x: self._tracked_objects[id(x[0])][3]
                    )[:10]
                ]
            }
        
    def _count_by_type(self) -> Dict[str, int]:
        """Count tracked objects by type."""
        counts = {}
        for obj_id, (ref, description, _, _) in self._tracked_objects.items():
            obj = ref()
            if obj is not None:
                type_name = type(obj).__name__
                counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def collect_garbage(self) -> Tuple[int, int, int]:
        """
        Force garbage collection.
        
        Returns:
            Tuple of (collected, uncollectable, collected_by_generation)
        """
        return gc.collect()
        
    def get_garbage_objects(self) -> List[Any]:
        """
        Get objects that cannot be collected by the garbage collector.
        
        Returns:
            List of uncollectable objects
        """
        return gc.garbage
        
    def print_object_referrers(self, obj: Any, max_depth: int = 3) -> None:
        """
        Print objects that reference the given object.
        
        Args:
            obj: Object to find referrers for
            max_depth: Maximum recursion depth
        """
        visited = set()
        
        def _print_referrers(o, depth=0, path=""):
            if depth >= max_depth or id(o) in visited:
                return
                
            visited.add(id(o))
            referrers = gc.get_referrers(o)
            
            for i, ref in enumerate(referrers):
                ref_type = type(ref).__name__
                ref_id = id(ref)
                ref_path = f"{path} -> {ref_type}({ref_id})"
                
                logger.info(f"{'  ' * depth}{ref_type}: {ref}")
                
                # Skip common built-in types
                if ref_type not in ('dict', 'list', 'tuple', 'set', 'function', 'frame'):
                    _print_referrers(ref, depth + 1, ref_path)
                    
        logger.info(f"Referrers for {type(obj).__name__}({id(obj)}):")
        _print_referrers(obj)


# Create a global resource tracker instance
resource_tracker = ResourceTracker()


class ResourceTracked:
    """
    Mixin for classes that want to be tracked by the resource tracker.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resource_tracker.track(self)
        
    def __del__(self):
        resource_tracker.untrack(self)


def track_resources(func):
    """
    Decorator for tracking resources used by a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Track memory usage before
        before = resource_tracker.get_memory_usage()
        
        # Call function
        result = func(*args, **kwargs)
        
        # Track memory usage after
        after = resource_tracker.get_memory_usage()
        
        # Log memory usage
        delta_mb = (after['rss'] - before['rss']) / (1024 * 1024)
        logger.debug(
            f"{func.__name__} memory usage: {delta_mb:.2f} MB "
            f"({before['rss_mb']:.2f} MB -> {after['rss_mb']:.2f} MB)"
        )
        
        return result
        
    return wrapper
