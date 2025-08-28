# utils/batch_processor.py
"""
Batch processing utilities for the Universal Translation System.
This module provides tools for efficient batch processing of data.
"""

import asyncio
import logging
import threading
import time
from typing import TypeVar, Generic, List, Callable, Awaitable, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


@dataclass
class BatchItem(Generic[T, R]):
    """An item in a batch with its result."""
    item: T
    result: Optional[R] = None
    error: Optional[Exception] = None
    future: Optional[asyncio.Future] = None
    priority: int = 0


class BatchProcessor(Generic[T, R]):
    """
    Processor for batching individual items into efficient batches.
    Automatically processes items when the batch is full or after a timeout.
    """
    
    def __init__(
        self,
        process_func: Callable[[List[T]], List[R]],
        batch_size: int = 32,
        timeout: float = 0.1,
        max_workers: int = 1,
        name: str = "BatchProcessor"
    ):
        """
        Initialize batch processor.
        
        Args:
            process_func: Function that processes a batch of items
            batch_size: Maximum batch size
            timeout: Maximum time to wait before processing a partial batch
            max_workers: Maximum number of worker threads
            name: Name for logging
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_workers = max_workers
        self.name = name
        
        self._batch: List[BatchItem[T, R]] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        
        logger.debug(
            f"Initialized {self.name} with batch_size={batch_size}, "
            f"timeout={timeout}, max_workers={max_workers}"
        )
        
    def process(self, item: T) -> R:
        """
        Process an item (blocking).
        
        Args:
            item: Item to process
            
        Returns:
            Processing result
            
        Raises:
            RuntimeError: If the processor is closed
            Exception: If processing fails
        """
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
            
        # Create batch item with a future
        batch_item = BatchItem(item=item)
        
        with self._lock:
            # Add to batch
            self._batch.append(batch_item)
            
            # Process batch if full
            if len(self._batch) >= self.batch_size:
                self._process_batch()
            # Start timer if this is the first item
            elif len(self._batch) == 1:
                self._start_timer()
                
        # Wait for result
        while batch_item.result is None and batch_item.error is None:
            time.sleep(0.01)
            
        # Raise error if processing failed
        if batch_item.error:
            raise batch_item.error
            
        return batch_item.result
        
    async def process_async(self, item: T) -> R:
        """
        Process an item asynchronously.
        
        Args:
            item: Item to process
            
        Returns:
            Processing result
            
        Raises:
            RuntimeError: If the processor is closed
            Exception: If processing fails
        """
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
            
        # Create batch item with a future
        future = asyncio.get_event_loop().create_future()
        batch_item = BatchItem(item=item, future=future)
        
        with self._lock:
            # Add to batch
            self._batch.append(batch_item)
            
            # Process batch if full
            if len(self._batch) >= self.batch_size:
                self._process_batch()
            # Start timer if this is the first item
            elif len(self._batch) == 1:
                self._start_timer()
                
        # Wait for result
        return await future
        
    def _start_timer(self):
        """Start the timeout timer."""
        if self._timer:
            self._timer.cancel()
            
        self._timer = threading.Timer(self.timeout, self._timeout_callback)
        self._timer.daemon = True
        self._timer.start()
        
    def _timeout_callback(self):
        """Called when the timeout expires."""
        with self._lock:
            if self._batch:
                logger.debug(
                    f"{self.name} processing batch of {len(self._batch)} "
                    f"items due to timeout"
                )
                self._process_batch()
                
    def _process_batch(self):
        """Process the current batch."""
        # Sort batch by priority (higher first)
        self._batch.sort(key=lambda item: item.priority, reverse=True)
        if not self._batch:
            return
            
        # Get the current batch
        batch = self._batch
        self._batch = []
        
        # Cancel the timer
        if self._timer:
            self._timer.cancel()
            self._timer = None
            
        # Process the batch in a worker thread
        self._executor.submit(self._process_batch_worker, batch)
        
    def _process_batch_worker(self, batch: List[BatchItem[T, R]]):
        """Worker function for processing a batch."""
        try:
            # Extract items
            items = [item.item for item in batch]
            
            # Process batch
            start_time = time.time()
            results = self.process_func(items)
            elapsed = time.time() - start_time
            
            # Check result length
            if len(results) != len(items):
                error = ValueError(
                    f"Expected {len(items)} results, got {len(results)}"
                )
                for item in batch:
                    item.error = error
                    if item.future:
                        item.future.set_exception(error)
                return
                
            # Set results
            for i, result in enumerate(results):
                batch[i].result = result
                if batch[i].future:
                    batch[i].future.set_result(result)
                    
            logger.debug(
                f"{self.name} processed batch of {len(batch)} items "
                f"in {elapsed:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Set error for all items
            for item in batch:
                item.error = e
                if item.future:
                    item.future.set_exception(e)
                    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        with self._lock:
           return {
               "processed_items": self._processed_items,
               "failed_items": self._failed_items,
               "total_batches": self._total_batches,
               "current_batch_size": len(self._batch),
               "avg_processing_time": (
                   self._total_processing_time / self._total_batches 
                   if self._total_batches > 0 else 0
                )
            }

    # Add to BatchProcessor class
    def process_with_retry(self, item: T, max_retries: int = 3, 
                          retry_delay: float = 0.5) -> R:
        """Process an item with automatic retries."""
        retries = 0
        last_error = None
    
        while retries <= max_retries:
            try:
                return self.process(item)
            except Exception as e:
                last_error = e
                retries += 1
            
                if retries <= max_retries:
                    time.sleep(retry_delay * (2 ** (retries - 1)))  # Exponential backoff
                
        if last_error:
            raise last_error

    def flush(self):
        """Process any remaining items in the batch."""
        with self._lock:
            if self._batch:
                logger.debug(
                    f"{self.name} flushing batch of {len(self._batch)} items"
                )
                self._process_batch()
                
    def close(self):
        """Close the processor and release resources."""
        self.flush()
        self._closed = True
        self._executor.shutdown()
        
        if self._timer:
            self._timer.cancel()
            self._timer = None
            
        logger.debug(f"{self.name} closed")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncBatchProcessor(Generic[T, R]):
    """
    Asynchronous processor for batching individual items into efficient batches.
    Automatically processes items when the batch is full or after a timeout.
    """
    
    def __init__(
        self,
        process_func: Callable[[List[T]], Awaitable[List[R]]],
        batch_size: int = 32,
        timeout: float = 0.1,
        name: str = "AsyncBatchProcessor"
    ):
        """
        Initialize async batch processor.
        
        Args:
            process_func: Async function that processes a batch of items
            batch_size: Maximum batch size
            timeout: Maximum time to wait before processing a partial batch
            name: Name for logging
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.timeout = timeout
        self.name = name
        
        self._batch: List[Tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None
        self._closed = False
        
        logger.debug(
            f"Initialized {self.name} with batch_size={batch_size}, "
            f"timeout={timeout}"
        )
        
    async def process(self, item: T) -> R:
        """
        Process an item asynchronously.
        
        Args:
            item: Item to process
            
        Returns:
            Processing result
            
        Raises:
            RuntimeError: If the processor is closed
            Exception: If processing fails
        """
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
            
        # Create future for this item
        future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            # Add to batch
            self._batch.append((item, future))
            
            # Process batch if full
            if len(self._batch) >= self.batch_size:
                await self._process_batch()
            # Start timer if this is the first item
            elif len(self._batch) == 1:
                await self._start_timer()
                
        # Wait for result
        return await future
        
    async def _start_timer(self):
        """Start the timeout timer."""
        if self._timer_task:
            self._timer_task.cancel()
            
        self._timer_task = asyncio.create_task(self._timeout_callback())
        
    async def _timeout_callback(self):
        """Called when the timeout expires."""
        await asyncio.sleep(self.timeout)
        
        async with self._lock:
            if self._batch:
                logger.debug(
                    f"{self.name} processing batch of {len(self._batch)} "
                    f"items due to timeout"
                )
                await self._process_batch()
                
    async def _process_batch(self):
        """Process the current batch."""
        if not self._batch:
            return
            
        # Get the current batch
        batch = self._batch
        self._batch = []
        
        # Cancel the timer
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
            
        # Process the batch
        try:
            # Extract items
            items = [item for item, _ in batch]
            futures = [future for _, future in batch]
            
            # Process batch
            start_time = time.time()
            results = await self.process_func(items)
            elapsed = time.time() - start_time
            
            # Check result length
            if len(results) != len(items):
                error = ValueError(
                    f"Expected {len(items)} results, got {len(results)}"
                )
                for future in futures:
                    future.set_exception(error)
                return
                
            # Set results
            for i, result in enumerate(results):
                futures[i].set_result(result)
                
            logger.debug(
                f"{self.name} processed batch of {len(batch)} items "
                f"in {elapsed:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Set error for all items
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
                    
    async def flush(self):
        """Process any remaining items in the batch."""
        async with self._lock:
            if self._batch:
                logger.debug(
                    f"{self.name} flushing batch of {len(self._batch)} items"
                )
                await self._process_batch()
                
    async def close(self):
        """Close the processor and release resources."""
        await self.flush()
        self._closed = True
        
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
            
        logger.debug(f"{self.name} closed")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def batch_decorator(
    batch_size: int = 32,
    timeout: float = 0.1,
    max_workers: int = 1
):
    """
    Decorator for automatically batching function calls.
    
    Args:
        batch_size: Maximum batch size
        timeout: Maximum time to wait before processing a partial batch
        max_workers: Maximum number of worker threads
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create a batch processor for this function
        processor = None
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal processor
            
            # Initialize processor if needed
            if processor is None:
                processor = BatchProcessor(
                    process_func=func,
                    batch_size=batch_size,
                    timeout=timeout,
                    max_workers=max_workers,
                    name=f"BatchProcessor({func.__name__})"
                )
                
            # Process the item
            return processor.process(args[0] if args else kwargs)
            
        # Add management methods
        wrapper.flush = lambda: processor.flush() if processor else None
        wrapper.close = lambda: processor.close() if processor else None
        
        return wrapper
        
    return decorator


def async_batch_decorator(
    batch_size: int = 32,
    timeout: float = 0.1
):
    """
    Decorator for automatically batching async function calls.
    
    Args:
        batch_size: Maximum batch size
        timeout: Maximum time to wait before processing a partial batch
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create a batch processor for this function
        processor = None
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal processor
            
            # Initialize processor if needed
            if processor is None:
                processor = AsyncBatchProcessor(
                    process_func=func,
                    batch_size=batch_size,
                    timeout=timeout,
                    name=f"AsyncBatchProcessor({func.__name__})"
                )
                
            # Process the item
            return await processor.process(args[0] if args else kwargs)
            
        # Add management methods
        wrapper.flush = lambda: processor.flush() if processor else None
        wrapper.close = lambda: processor.close() if processor else None
        
        return wrapper
        
    return decorator
