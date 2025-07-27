"""
Common utilities and shared components.

This module contains shared utilities that can be used across
different layers of the application without creating circular dependencies.
"""

from .observability import *
from .logging import *

__all__ = [
    # Core observability
    "get_observable_logger",
    "with_observability",
    "get_current_trace_id", 
    "get_current_span_id",
    "create_trace_context",
    "TraceContext",
    "TraceContextManager",
    
    # Setup and configuration
    "setup_observability",
    "quick_setup",
    "production_setup", 
    "development_setup",
    
    # Logging configuration
    "LogLevel",
    "LoggingConfig",
    "configure_structured_logging",
    
    # Integration
    "initialize_observability",
    "get_observability_integration",
]