# utils/dependency_container.py
"""
Lightweight dependency injection container for the Universal Translation System.
This module provides a centralized way to manage dependencies and their lifecycles.
"""

import inspect
from typing import Dict, Any, Callable, Optional, Type, TypeVar, cast
import logging
from functools import wraps
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DependencyContainer:
    """
    A lightweight dependency injection container that manages
    component lifecycles and dependencies.
    """
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._singletons: Dict[str, bool] = {}
        
    def register(self, name: str, factory: Callable[..., T], singleton: bool = True) -> None:
        """
        Register a component factory with the container.
        
        Args:
            name: Unique name for the component
            factory: Factory function that creates the component
            singleton: If True, the component will be created only once
        """
        if name in self._factories:
            logger.warning(f"Overwriting existing component: {name}")
            
        self._factories[name] = factory
        self._singletons[name] = singleton
        
        # Clear instance if it exists
        if name in self._components:
            del self._components[name]
            
        logger.debug(f"Registered component: {name} (singleton={singleton})")
    
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register an existing instance with the container.
        
        Args:
            name: Unique name for the component
            instance: The component instance
        """
        self._components[name] = instance
        self._singletons[name] = True
        logger.debug(f"Registered instance: {name}")
    
    def register_class(self, cls: Type[T], name: Optional[str] = None, singleton: bool = True) -> None:
        """
        Register a class with the container.
        
        Args:
            cls: The class to register
            name: Optional name (defaults to class name)
            singleton: If True, the component will be created only once
        """
        component_name = name or cls.__name__
        
        # Create a factory that will resolve constructor dependencies
        def factory(**kwargs):
            try:
                # Get constructor signature
                sig = inspect.signature(cls.__init__)
                params = {}
                
                # For each parameter, try to resolve it from the container
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    if param_name in kwargs:
                        params[param_name] = kwargs[param_name]
                    elif param.annotation != inspect.Parameter.empty:
                        # Try to resolve by type annotation
                        type_name = param.annotation.__name__
                        if self.has(type_name):
                            params[param_name] = self.resolve(type_name)
                        elif param.default != inspect.Parameter.empty:
                            params[param_name] = param.default
                        else:
                            raise ConfigurationError(
                                f"Cannot resolve parameter '{param_name}' for {component_name}"
                            )
                    elif param.default != inspect.Parameter.empty:
                        params[param_name] = param.default
                    else:
                        raise ConfigurationError(
                            f"Cannot resolve parameter '{param_name}' for {component_name}"
                        )
                
                return cls(**params)
            except Exception as e:
                logger.error(f"Error creating {component_name}: {str(e)}")
                raise
        
        self.register(component_name, factory, singleton)
    
    def has(self, name: str) -> bool:
        """
        Check if a component is registered.
        
        Args:
            name: Component name
            
        Returns:
            True if the component is registered
        """
        return name in self._factories or name in self._components
    
    def resolve(self, name: str, **kwargs) -> Any:
        """
        Resolve a component by name.
        
        Args:
            name: Component name
            **kwargs: Override dependencies
            
        Returns:
            The component instance
            
        Raises:
            ConfigurationError: If the component is not registered
        """
        # Return existing instance for singletons
        if name in self._components and self._singletons.get(name, False):
            return self._components[name]
            
        # Create new instance
        if name in self._factories:
            factory = self._factories[name]
            instance = factory(**kwargs)
            
            # Cache singleton instances
            if self._singletons.get(name, False):
                self._components[name] = instance
                
            return instance
        
        raise ConfigurationError(f"Component not registered: {name}")
    
    def resolve_all(self, **kwargs) -> Dict[str, Any]:
        """
        Resolve all registered components.
        
        Args:
            **kwargs: Override dependencies
            
        Returns:
            Dictionary of component instances
        """
        result = {}
        for name in self._factories:
            result[name] = self.resolve(name, **kwargs)
        return result
    
    def clear(self) -> None:
        """Clear all cached component instances."""
        self._components.clear()
        logger.debug("Cleared all component instances")
    
    def remove(self, name: str) -> None:
        """
        Remove a component from the container.
        
        Args:
            name: Component name
        """
        if name in self._factories:
            del self._factories[name]
        if name in self._singletons:
            del self._singletons[name]
        if name in self._components:
            del self._components[name]
        logger.debug(f"Removed component: {name}")


# Create a global container instance
container = DependencyContainer()


def inject(**dependencies):
    """
    Decorator for injecting dependencies into functions.
    
    Args:
        **dependencies: Dependencies to inject
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve dependencies
            for name, component_name in dependencies.items():
                if name not in kwargs:
                    kwargs[name] = container.resolve(component_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def injectable(cls=None, *, singleton: bool = True):
    """
    Decorator for making a class injectable.
    
    Args:
        cls: The class to make injectable
        singleton: If True, the class will be a singleton
        
    Returns:
        Decorated class
    """
    def decorator(cls):
        container.register_class(cls, singleton=singleton)
        return cls
        
    if cls is None:
        return decorator
    return decorator(cls)
