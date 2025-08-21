"""
Circuit breaker pattern implementation for decoder nodes.
This helps prevent cascading failures and allows for graceful degradation.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Any, Dict, Optional, List, Tuple
import traceback

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, don't try
    HALF_OPEN = "HALF_OPEN"  # Testing if working again

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for decoder nodes.
    
    The circuit breaker pattern prevents cascading failures by failing fast
    when a service is unavailable. It has three states:
    
    - CLOSED: Normal operation, requests are passed through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if the service is working again
    
    When in the OPEN state, the circuit breaker will periodically allow a request
    through to test if the service has recovered. If the request succeeds, the
    circuit breaker returns to the CLOSED state. If it fails, it remains OPEN.
    """
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        timeout: int = 10,
        half_open_success_threshold: int = 3
    ):
        """
        Initialize a circuit breaker.
        
        Args:
            name: Name of the circuit (for logging)
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            timeout: Seconds to wait before timing out a request
            half_open_success_threshold: Number of successful requests needed to close the circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.half_open_success_threshold = half_open_success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.error_counts: Dict[str, int] = {}  # Track error types
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            Exception: If the circuit is open or the function fails
        """
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                logger.warning(f"Circuit '{self.name}' is OPEN, request rejected")
                raise CircuitBreakerOpenError(f"Circuit '{self.name}' is open")
        
        # Execute with timeout
        try:
            # Create a task with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            
            # Record success
            self._record_success()
            return result
            
        except asyncio.TimeoutError:
            self._record_failure("timeout")
            raise TimeoutError(f"Request to '{self.name}' timed out after {self.timeout}s")
            
        except Exception as e:
            self._record_failure(type(e).__name__)
            logger.error(f"Circuit '{self.name}' request failed: {str(e)}")
            raise
    
    def _record_success(self):
        """Record a successful request"""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"Circuit '{self.name}' successful request in HALF_OPEN state ({self.success_count}/{self.half_open_success_threshold})")
            
            if self.success_count >= self.half_open_success_threshold:
                logger.info(f"Circuit '{self.name}' transitioning from HALF_OPEN to CLOSED")
                self.reset()
    
    def _record_failure(self, error_type: str):
        """
        Record a failed request
        
        Args:
            error_type: Type of error that occurred
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Track error types
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit '{self.name}' transitioning from CLOSED to OPEN after {self.failure_count} failures")
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit '{self.name}' transitioning from HALF_OPEN to OPEN after failure")
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Reset the circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.error_counts = {}
        logger.info(f"Circuit '{self.name}' reset to CLOSED state")
        
    @property
    def is_available(self) -> bool:
        """Check if the circuit is available for requests"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            return True
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "error_counts": self.error_counts,
            "is_available": self.is_available
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when a circuit is open"""
    pass

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    Useful for tracking all decoder nodes in the coordinator.
    """
    
    def __init__(self):
        """Initialize the circuit breaker registry"""
        self.circuits: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        timeout: int = 10
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before trying again
            timeout: Seconds to wait before timing out a request
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuits:
            self.circuits[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout
            )
        return self.circuits[name]
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return [circuit.get_stats() for circuit in self.circuits.values()]
    
    def get_available_circuits(self) -> List[Tuple[str, CircuitBreaker]]:
        """Get all available circuits"""
        return [(name, circuit) for name, circuit in self.circuits.items() if circuit.is_available]
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for circuit in self.circuits.values():
            circuit.reset()
    
    def remove(self, name: str):
        """Remove a circuit breaker from the registry"""
        if name in self.circuits:
            del self.circuits[name]

# Global registry instance
registry = CircuitBreakerRegistry()