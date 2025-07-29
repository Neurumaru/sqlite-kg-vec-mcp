"""
Trace and span context management for observability.
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    """
    Contains trace and span information for observability.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation: Optional[str] = None
    layer: Optional[str] = None
    component: Optional[str] = None
    start_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trace context."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "layer": self.layer,
            "component": self.component,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "metadata": self.metadata,
        }


# Context variables for trace propagation
_trace_context: ContextVar[Optional[TraceContext]] = ContextVar("trace_context", default=None)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID from context.

    Returns:
        Current trace ID or None if no trace is active
    """
    context = _trace_context.get()
    return context.trace_id if context else None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID from context.

    Returns:
        Current span ID or None if no trace is active
    """
    context = _trace_context.get()
    return context.span_id if context else None


def get_current_trace_context() -> Optional[TraceContext]:
    """
    Get the current trace context.

    Returns:
        Current trace context or None if no trace is active
    """
    return _trace_context.get()


def create_trace_context(
    operation: str,
    layer: str,
    component: str,
    parent_context: Optional[TraceContext] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TraceContext:
    """
    Create a new trace context.

    Args:
        operation: Operation name
        layer: Layer name (domain, port, adapter)
        component: Component name
        parent_context: Parent trace context
        metadata: Additional metadata

    Returns:
        New trace context
    """
    if parent_context:
        trace_id = parent_context.trace_id
        parent_span_id = parent_context.span_id
    else:
        trace_id = str(uuid.uuid4())
        parent_span_id = None

    span_id = str(uuid.uuid4())

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation=operation,
        layer=layer,
        component=component,
        metadata=metadata or {},
    )


def set_trace_context(context: Optional[TraceContext]) -> None:
    """
    Set the current trace context.

    Args:
        context: Trace context to set
    """
    _trace_context.set(context)


class TraceContextManager:
    """
    Context manager for trace contexts.
    """

    def __init__(self, trace_context: TraceContext):
        """
        Initialize context manager.

        Args:
            trace_context: Trace context to use
        """
        self.trace_context = trace_context
        self.previous_context: Optional[TraceContext] = None

    def __enter__(self) -> TraceContext:
        """Enter the context."""
        self.previous_context = _trace_context.get()
        _trace_context.set(self.trace_context)
        return self.trace_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        _trace_context.set(self.previous_context)


def with_trace_context(trace_context: TraceContext):
    """
    Decorator to run function with a specific trace context.

    Args:
        trace_context: Trace context to use

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with TraceContextManager(trace_context):
                return func(*args, **kwargs)

        return wrapper

    return decorator
