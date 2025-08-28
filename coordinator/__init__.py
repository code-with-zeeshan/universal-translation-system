# coordinator/__init__.py
"""
Coordinator package for managing decoder pools, health checks, and routing.

Exposes circuit breaker utilities. Import advanced_coordinator directly if you
need the FastAPI app or pool runtime components to avoid import side effects.
"""
from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
]