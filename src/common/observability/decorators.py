"""
Decorators for automatic observability integration.
"""

import functools
import inspect
from typing import Optional, Dict, Any, Callable, Union

from .context import (
    create_trace_context,
    TraceContextManager,
    get_current_trace_context
)
from .logger import get_observable_logger


def with_observability(
    operation: Optional[str] = None,
    layer: Optional[str] = None,
    component: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False
):
    """
    Decorator to add automatic observability to functions.
    
    This decorator:
    - Creates trace context if none exists
    - Logs operation start/completion/failure
    - Measures execution time
    - Handles exceptions with structured logging
    
    Args:
        operation: Operation name (defaults to function name)
        layer: Layer name (tries to infer from module)
        component: Component name (tries to infer from class/module)
        include_args: Whether to log function arguments
        include_result: Whether to log function result
    """
    def decorator(func: Callable) -> Callable:
        # Infer metadata if not provided
        func_operation = operation or func.__name__
        func_layer = layer or _infer_layer(func)
        func_component = component or _infer_component(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create trace context
            current_context = get_current_trace_context()
            if current_context is None:
                trace_context = create_trace_context(
                    operation=func_operation,
                    layer=func_layer,
                    component=func_component
                )
                use_context_manager = True
            else:
                # Create child span
                trace_context = create_trace_context(
                    operation=func_operation,
                    layer=func_layer,
                    component=func_component,
                    parent_context=current_context
                )
                use_context_manager = True
            
            # Get logger
            logger = get_observable_logger(func_component, func_layer)
            
            # Prepare logging context
            log_context = {}
            if include_args and args:
                log_context["args"] = _sanitize_args(args)
            if include_args and kwargs:
                log_context["kwargs"] = _sanitize_kwargs(kwargs)
            
            def execute_function():
                # Log operation start
                start_time = logger.operation_started(func_operation, **log_context)
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Log success
                    success_context = log_context.copy()
                    if include_result and result is not None:
                        success_context["result"] = _sanitize_result(result)
                    
                    logger.operation_completed(func_operation, start_time, **success_context)
                    
                    return result
                    
                except Exception as e:
                    # Log failure
                    logger.operation_failed(func_operation, start_time, e, **log_context)
                    raise
            
            # Execute with or without trace context
            if use_context_manager:
                with TraceContextManager(trace_context):
                    return execute_function()
            else:
                return execute_function()
        
        return wrapper
    return decorator


def with_trace(
    operation: Optional[str] = None,
    layer: Optional[str] = None,
    component: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator to add trace context to functions.
    
    Args:
        operation: Operation name
        layer: Layer name
        component: Component name  
        metadata: Additional metadata
    """
    def decorator(func: Callable) -> Callable:
        func_operation = operation or func.__name__
        func_layer = layer or _infer_layer(func)
        func_component = component or _infer_component(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_context = get_current_trace_context()
            trace_context = create_trace_context(
                operation=func_operation,
                layer=func_layer,
                component=func_component,
                parent_context=current_context,
                metadata=metadata
            )
            
            with TraceContextManager(trace_context):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_metrics(
    metric_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
    """
    Decorator to add automatic metrics collection.
    
    Args:
        metric_name: Metric name (defaults to function name)
        tags: Additional metric tags
    """
    def decorator(func: Callable) -> Callable:
        func_metric_name = metric_name or f"{func.__name__}_calls"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger with observability service
            component = _infer_component(func)
            layer = _infer_layer(func)
            logger = get_observable_logger(component, layer)
            
            # Record metric
            if logger.observability_service and hasattr(logger.observability_service, 'record_metric'):
                metric_tags = {
                    "layer": layer,
                    "component": component,
                    "function": func.__name__
                }
                if tags:
                    metric_tags.update(tags)
                
                logger.observability_service.record_metric(
                    func_metric_name,
                    1,
                    tags=metric_tags
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def _infer_layer(func: Callable) -> str:
    """Infer layer from function module path."""
    module = inspect.getmodule(func)
    if module and module.__name__:
        module_path = module.__name__
        if 'adapters' in module_path:
            return 'adapter'
        elif 'ports' in module_path:
            return 'port'
        elif 'domain' in module_path:
            return 'domain'
        elif 'application' in module_path:
            return 'application'
    return 'unknown'


def _infer_component(func: Callable) -> str:
    """Infer component from function context."""
    # Try to get class name if it's a method
    if hasattr(func, '__self__'):
        return func.__self__.__class__.__name__.lower()
    
    # Get from module name
    module = inspect.getmodule(func)
    if module and module.__name__:
        parts = module.__name__.split('.')
        if len(parts) > 0:
            return parts[-1]
    
    return func.__name__


def _sanitize_args(args: tuple) -> list:
    """Sanitize function arguments for logging."""
    sanitized = []
    for arg in args:
        if hasattr(arg, '__dict__'):
            # Object - just include type name
            sanitized.append(f"<{type(arg).__name__}>")
        elif isinstance(arg, (str, int, float, bool, type(None))):
            sanitized.append(arg)
        else:
            sanitized.append(f"<{type(arg).__name__}>")
    return sanitized


def _sanitize_kwargs(kwargs: dict) -> dict:
    """Sanitize function keyword arguments for logging."""
    sanitized = {}
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        else:
            sanitized[key] = f"<{type(value).__name__}>"
    return sanitized


def _sanitize_result(result: Any) -> Any:
    """Sanitize function result for logging."""
    if isinstance(result, (str, int, float, bool, type(None))):
        return result
    elif hasattr(result, '__dict__'):
        return f"<{type(result).__name__}>"
    else:
        return f"<{type(result).__name__}>"