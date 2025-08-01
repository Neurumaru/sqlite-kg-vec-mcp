"""
Observability utilities for logging, tracing, and metrics.
"""

from .context import (
    TraceContext,
    TraceContextManager,
    create_trace_context,
    get_current_span_id,
    get_current_trace_id,
    set_trace_context,
)
from .decorators import (
    with_metrics,
    with_observability,
    with_trace,
)
from .integration import (
    ObservabilityIntegration,
    get_observability_integration,
    initialize_observability,
)
from .logger import (
    ObservableLogger,
    get_observable_logger,
)
from .setup import (
    development_setup,
    production_setup,
    quick_setup,
    setup_observability,
)

__all__ = [
    # Context management
    "get_current_trace_id",
    "get_current_span_id",
    "create_trace_context",
    "set_trace_context",
    "TraceContext",
    "TraceContextManager",
    # Logging
    "ObservableLogger",
    "get_observable_logger",
    # Decorators
    "with_observability",
    "with_trace",
    "with_metrics",
    # Integration
    "ObservabilityIntegration",
    "initialize_observability",
    "get_observability_integration",
    # Setup
    "setup_observability",
    "quick_setup",
    "production_setup",
    "development_setup",
]
