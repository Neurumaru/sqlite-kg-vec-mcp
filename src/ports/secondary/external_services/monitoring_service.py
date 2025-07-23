"""
Monitoring service port for observability and telemetry.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class MetricType(Enum):
    """Types of metrics that can be recorded."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringService(ABC):
    """
    Secondary port for monitoring and observability operations.

    This interface defines how the domain interacts with monitoring services
    for logging, metrics, tracing, and alerting.
    """

    # Metrics
    @abstractmethod
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for the metric
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if metric was recorded successfully
        """
        pass

    @abstractmethod
    async def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Increment value (default: 1.0)
            tags: Optional tags

        Returns:
            True if counter was incremented successfully
        """
        pass

    @abstractmethod
    async def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Set a gauge metric value.

        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags

        Returns:
            True if gauge was set successfully
        """
        pass

    @abstractmethod
    async def record_timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Record a timing metric.

        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            tags: Optional tags

        Returns:
            True if timing was recorded successfully
        """
        pass

    # Logging
    @abstractmethod
    async def log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Log a message with structured data.

        Args:
            level: Log level
            message: Log message
            extra: Optional extra data
            tags: Optional tags

        Returns:
            True if log was recorded successfully
        """
        pass

    @abstractmethod
    async def log_event(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Log a structured event.

        Args:
            event_name: Name of the event
            event_data: Event data
            tags: Optional tags

        Returns:
            True if event was logged successfully
        """
        pass

    @abstractmethod
    async def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Log an error with context information.

        Args:
            error: Exception that occurred
            context: Optional context information
            tags: Optional tags

        Returns:
            True if error was logged successfully
        """
        pass

    # Tracing
    @abstractmethod
    async def start_trace(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new trace span.

        Args:
            operation_name: Name of the operation
            trace_id: Optional trace ID (will be generated if not provided)
            parent_span_id: Optional parent span ID
            tags: Optional tags for the span

        Returns:
            Span ID for the created span
        """
        pass

    @abstractmethod
    async def finish_trace(
        self,
        span_id: str,
        status: str = "ok",
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Finish a trace span.

        Args:
            span_id: ID of the span to finish
            status: Status of the operation ("ok", "error", etc.)
            tags: Optional additional tags

        Returns:
            True if span was finished successfully
        """
        pass

    @abstractmethod
    async def add_trace_annotation(
        self,
        span_id: str,
        key: str,
        value: Any
    ) -> bool:
        """
        Add an annotation to a trace span.

        Args:
            span_id: ID of the span
            key: Annotation key
            value: Annotation value

        Returns:
            True if annotation was added successfully
        """
        pass

    # Search-specific monitoring
    @abstractmethod
    async def track_search_request(
        self,
        query: str,
        search_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Track a search request for analytics.

        Args:
            query: Search query
            search_type: Type of search performed
            user_id: Optional user ID
            session_id: Optional session ID

        Returns:
            Search request ID for tracking
        """
        pass

    @abstractmethod
    async def track_search_results(
        self,
        request_id: str,
        result_count: int,
        response_time_ms: float,
        relevance_scores: Optional[List[float]] = None
    ) -> bool:
        """
        Track search results and performance.

        Args:
            request_id: Search request ID
            result_count: Number of results returned
            response_time_ms: Response time in milliseconds
            relevance_scores: Optional relevance scores

        Returns:
            True if tracking was successful
        """
        pass

    @abstractmethod
    async def track_user_interaction(
        self,
        user_id: str,
        action: str,
        entity_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track user interactions with the system.

        Args:
            user_id: User ID
            action: Action performed
            entity_id: Optional entity ID involved
            metadata: Optional metadata

        Returns:
            True if interaction was tracked successfully
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def record_health_check(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a health check result.

        Args:
            component: Component name
            status: Health status ("healthy", "unhealthy", "degraded")
            details: Optional health details

        Returns:
            True if health check was recorded successfully
        """
        pass

    @abstractmethod
    async def get_metrics_summary(
        self,
        metric_names: Optional[List[str]] = None,
        time_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get a summary of metrics over a time range.

        Args:
            metric_names: Optional list of specific metrics
            time_range_minutes: Time range in minutes

        Returns:
            Metrics summary
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the monitoring service itself.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def flush_metrics(self) -> bool:
        """
        Flush any buffered metrics to the backend.

        Returns:
            True if flush was successful
        """
        pass
