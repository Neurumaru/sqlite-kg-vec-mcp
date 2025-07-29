"""
Common utilities and shared components.

This module contains shared utilities that can be used across
different layers of the application without creating circular dependencies.
"""

from .logging import (
    LoggingConfig,
    LogLevel,
    ObservableLogger,
    configure_structured_logging,
    get_observable_logger,
)
from .observability import (
    TraceContext,
    TraceContextManager,
    create_trace_context,
    development_setup,
    get_current_span_id,
    get_current_trace_id,
    get_observability_integration,
    initialize_observability,
    production_setup,
    quick_setup,
    setup_observability,
    with_observability,
)

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
    "ObservableLogger",
    "configure_structured_logging",
    # Integration
    "initialize_observability",
    "get_observability_integration",
]
