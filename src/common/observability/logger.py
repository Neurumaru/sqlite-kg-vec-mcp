"""
Observable logger that integrates with trace context and structured logging.
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

import structlog

from .context import TraceContext, get_current_trace_context


class LogLevel(Enum):
    """Log levels enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ObservableLogger:
    """
    Logger that integrates with observability context and structured logging.

    This logger automatically includes trace information and provides
    consistent structured logging across all components.
    """

    def __init__(
        self, component: str, layer: str, observability_service: Optional[Any] = None
    ):
        """
        Initialize observable logger.

        Args:
            component: Component name (e.g., "sqlite_repository")
            layer: Layer name ("domain", "port", "adapter")
            observability_service: Optional observability service for metrics/tracing
        """
        self.component = component
        self.layer = layer
        self.observability_service = observability_service

        # Initialize underlying logger
        self.logger = structlog.get_logger(component)

    def _get_base_context(self) -> Dict[str, Any]:
        """Get base logging context with trace information."""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "layer": self.layer,
            "component": self.component,
        }

        # Add trace context if available
        trace_context = get_current_trace_context()
        if trace_context:
            context.update(
                {
                    "trace_id": trace_context.trace_id,
                    "span_id": trace_context.span_id,
                    "parent_span_id": trace_context.parent_span_id,
                    "operation": trace_context.operation,
                }
            )

        return context

    def _log(self, level: LogLevel, event: str, **kwargs) -> None:
        """
        Internal logging method.

        Args:
            level: Log level
            event: Event name/description
            **kwargs: Additional context
        """
        # Build log entry
        log_data = self._get_base_context()
        log_data["event"] = event
        log_data["level"] = level.value
        log_data.update(kwargs)

        # Log with structlog
        log_method = getattr(self.logger, level.value.lower())
        log_method(**log_data)

        # Send to observability service if available
        if self.observability_service and hasattr(
            self.observability_service, "log_event"
        ):
            trace_context = get_current_trace_context()
            if trace_context:
                self.observability_service.log_event(
                    span_id=trace_context.span_id, name=event, data=log_data
                )

    def debug(self, event: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, event, **kwargs)

    def exception_occurred(
        self, exception: Exception, operation: str, **kwargs
    ) -> None:
        """
        Log exception with rich context.

        Args:
            exception: Exception that occurred
            operation: Operation being performed
            **kwargs: Additional context
        """
        self.error(
            "exception_occurred",
            operation=operation,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            **kwargs,
        )

        # Record exception metric if observability service available
        if self.observability_service and hasattr(
            self.observability_service, "record_metric"
        ):
            self.observability_service.record_metric(
                "exception_count",
                1,
                tags={
                    "layer": self.layer,
                    "component": self.component,
                    "exception_type": type(exception).__name__,
                    "operation": operation,
                },
            )

    def operation_started(self, operation: str, **kwargs) -> float:
        """
        Log operation start and return start time.

        Args:
            operation: Operation name
            **kwargs: Additional context

        Returns:
            Start time for duration calculation
        """
        start_time = time.time()

        self.info(
            "operation_started", operation=operation, start_time=start_time, **kwargs
        )

        return start_time

    def operation_completed(self, operation: str, start_time: float, **kwargs) -> None:
        """
        Log operation completion with duration.

        Args:
            operation: Operation name
            start_time: Start time from operation_started
            **kwargs: Additional context
        """
        duration_ms = (time.time() - start_time) * 1000

        self.info(
            "operation_completed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

        # Record performance metric
        if self.observability_service and hasattr(
            self.observability_service, "record_metric"
        ):
            self.observability_service.record_metric(
                "operation_duration_ms",
                duration_ms,
                tags={
                    "layer": self.layer,
                    "component": self.component,
                    "operation": operation,
                },
            )

    def operation_failed(
        self, operation: str, start_time: float, exception: Exception, **kwargs
    ) -> None:
        """
        Log operation failure with duration and exception.

        Args:
            operation: Operation name
            start_time: Start time from operation_started
            exception: Exception that caused failure
            **kwargs: Additional context
        """
        duration_ms = (time.time() - start_time) * 1000

        self.error(
            "operation_failed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            **kwargs,
        )

        # Record failure metrics
        if self.observability_service:
            if hasattr(self.observability_service, "record_metric"):
                self.observability_service.record_metric(
                    "operation_failure_count",
                    1,
                    tags={
                        "layer": self.layer,
                        "component": self.component,
                        "operation": operation,
                        "exception_type": type(exception).__name__,
                    },
                )


# Global logger registry
_logger_registry: Dict[str, ObservableLogger] = {}


def get_observable_logger(
    component: str, layer: str, observability_service: Optional[Any] = None
) -> ObservableLogger:
    """
    Get or create an observable logger for a component.

    Args:
        component: Component name
        layer: Layer name
        observability_service: Optional observability service

    Returns:
        Observable logger instance
    """
    key = f"{layer}.{component}"

    if key not in _logger_registry:
        _logger_registry[key] = ObservableLogger(
            component=component,
            layer=layer,
            observability_service=observability_service,
        )

    return _logger_registry[key]


def configure_structured_logging() -> None:
    """
    Configure structured logging for the application.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
