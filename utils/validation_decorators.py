# utils/validation_decorators.py
"""
Validation decorators for the Universal Translation System.
This module provides decorators for validating API inputs.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
from pydantic import BaseModel, create_model, ValidationError
# Optional FastAPI imports for API-layer validation. For environments without FastAPI,
# we provide minimal shims to avoid hard dependency during smoke/dry-run.
try:
    from fastapi import Depends, Request, HTTPException, status  # type: ignore
except Exception:  # pragma: no cover
    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
    class status:  # type: ignore
        HTTP_400_BAD_REQUEST = 400
    def Depends(x):  # type: ignore
        return x
    class Request:  # type: ignore
        pass
import logging
from .exceptions import ValidationError as UTSValidationError
from .unified_validation import InputValidator

logger = logging.getLogger(__name__)

T = TypeVar('T')


def validate_request_body(model: Type[BaseModel]):
    """
    Decorator for validating request body against a Pydantic model.
    
    Args:
        model: Pydantic model class
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args or kwargs
            request = None
            for arg in args:
                if hasattr(arg, 'json') and callable(getattr(arg, 'json')):
                    request = arg
                    break
                    
            if request is None:
                request = kwargs.get('request')
                
            if request is None:
                logger.warning(f"No request found in {func.__name__}")
                return await func(*args, **kwargs)
                
            # Validate request body
            try:
                body = await request.json()
                validator = InputValidator()
                validated_data = validator.validate_model(body, model)
                
                # Replace request with validated data
                for i, arg in enumerate(args):
                    if arg is request:
                        args = list(args)
                        args[i] = validated_data
                        args = tuple(args)
                        break
                        
                if 'request' in kwargs and kwargs['request'] is request:
                    kwargs['request'] = validated_data
                    
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
                
        return wrapper
    return decorator


def validate_query_params(model: Type[BaseModel]):
    """
    Decorator for validating query parameters against a Pydantic model.
    
    Args:
        model: Pydantic model class
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args or kwargs
            request = None
            for arg in args:
                if hasattr(arg, 'query_params') and hasattr(arg.query_params, 'items'):
                    request = arg
                    break
                    
            if request is None:
                request = kwargs.get('request')
                
            if request is None:
                logger.warning(f"No request found in {func.__name__}")
                return await func(*args, **kwargs)
                
            # Validate query parameters
            try:
                params = dict(request.query_params)
                validator = InputValidator()
                validated_data = validator.validate_model(params, model)
                
                # Add validated data to kwargs
                kwargs['validated_params'] = validated_data
                
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
                
        return wrapper
    return decorator


def validate_path_params(model: Type[BaseModel]):
    """
    Decorator for validating path parameters against a Pydantic model.
    
    Args:
        model: Pydantic model class
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args or kwargs
            request = None
            for arg in args:
                if hasattr(arg, 'path_params') and hasattr(arg.path_params, 'items'):
                    request = arg
                    break
                    
            if request is None:
                request = kwargs.get('request')
                
            if request is None:
                logger.warning(f"No request found in {func.__name__}")
                return await func(*args, **kwargs)
                
            # Validate path parameters
            try:
                params = dict(request.path_params)
                validator = InputValidator()
                validated_data = validator.validate_model(params, model)
                
                # Add validated data to kwargs
                kwargs['validated_path_params'] = validated_data
                
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
                
        return wrapper
    return decorator


def validate_input(validator_func: Callable[[Any], Any]):
    """
    Decorator for validating function input with a custom validator.
    
    Args:
        validator_func: Function that validates the input
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get the first argument (excluding self/cls)
                if args and len(args) > 1:
                    validated_arg = validator_func(args[1])
                    args = list(args)
                    args[1] = validated_arg
                    args = tuple(args)
                    
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                if isinstance(e, UTSValidationError):
                    raise
                raise UTSValidationError(str(e))
                
        return wrapper
    return decorator


def validate_output(validator_func: Callable[[Any], Any]):
    """
    Decorator for validating function output with a custom validator.
    
    Args:
        validator_func: Function that validates the output
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                return validator_func(result)
            except Exception as e:
                logger.error(f"Output validation error in {func.__name__}: {e}")
                if isinstance(e, UTSValidationError):
                    raise
                raise UTSValidationError(str(e))
                
        return wrapper
    return decorator


def validate_model_input(model: Type[BaseModel], arg_name: Optional[str] = None):
    """
    Decorator for validating function input against a Pydantic model.
    
    Args:
        model: Pydantic model class
        arg_name: Name of the argument to validate (defaults to first argument)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = InputValidator()
            
            try:
                # Validate by argument name
                if arg_name is not None:
                    if arg_name in kwargs:
                        kwargs[arg_name] = validator.validate_model(kwargs[arg_name], model)
                    else:
                        # Try to find the argument position
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        
                        # Skip self/cls
                        if params and params[0] in ('self', 'cls'):
                            params = params[1:]
                            
                        if params and arg_name in params:
                            arg_index = params.index(arg_name)
                            if len(args) > arg_index:
                                args = list(args)
                                args[arg_index] = validator.validate_model(args[arg_index], model)
                                args = tuple(args)
                # Validate first argument
                elif args and len(args) > 1:
                    args = list(args)
                    args[1] = validator.validate_model(args[1], model)
                    args = tuple(args)
                    
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation error in {func.__name__}: {e}")
                if isinstance(e, UTSValidationError):
                    raise
                raise UTSValidationError(str(e))
                
        return wrapper
    return decorator
