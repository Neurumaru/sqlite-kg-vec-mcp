"""
Observability port interface.

This port defines how the domain interacts with observability systems
for tracing, logging, metrics collection, and performance monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ObservabilityService(ABC):
    """
    Port interface for observability and monitoring operations.
    
    This interface abstracts observability concerns, allowing the domain
    to work with different monitoring systems (Langfuse, OpenTelemetry, etc.)
    """

    @abstractmethod
    def create_trace(
        self,
        name: str,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new trace for tracking a business operation.
        
        Args:
            name: Human-readable name for the trace
            session_id: Session identifier
            user_id: Optional user identifier
            metadata: Additional metadata for the trace
            
        Returns:
            Trace identifier for referencing in subsequent operations
        """
        pass

    @abstractmethod
    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new span within a trace.
        
        Args:
            trace_id: Parent trace identifier
            name: Human-readable name for the span
            parent_span_id: Optional parent span identifier
            metadata: Additional metadata for the span
            
        Returns:
            Span identifier for referencing in subsequent operations
        """
        pass

    @abstractmethod
    def end_span(
        self,
        span_id: str,
        output: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None
    ) -> None:
        """
        End a span and record its output.
        
        Args:
            span_id: Span identifier
            output: Output data from the operation
            status: Operation status ('success', 'error', 'cancelled')
            error: Error message if status is 'error'
        """
        pass

    @abstractmethod
    def log_llm_generation(
        self,
        span_id: str,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        response: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an LLM generation event.
        
        Args:
            span_id: Parent span identifier
            model: Model identifier used for generation
            prompt: Input prompt (string or message format)
            response: Generated response
            usage: Token usage statistics
            metadata: Additional generation metadata
        """
        pass

    @abstractmethod
    def log_event(
        self,
        span_id: str,
        name: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log a custom event within a span.
        
        Args:
            span_id: Parent span identifier
            name: Event name
            data: Event data
            timestamp: Optional timestamp (defaults to current time)
        """
        pass

    @abstractmethod
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit of measurement
            tags: Optional tags for grouping/filtering
            timestamp: Optional timestamp (defaults to current time)
        """
        pass

    @abstractmethod
    def add_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a quality score to a trace.
        
        Args:
            trace_id: Target trace identifier
            name: Score name/type
            value: Numeric score value
            comment: Optional human-readable comment
            metadata: Additional score metadata
        """
        pass

    @abstractmethod
    def finalize_trace(
        self,
        trace_id: str,
        output: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None
    ) -> None:
        """
        Finalize a trace and record its final output.
        
        Args:
            trace_id: Trace identifier
            output: Final output data
            status: Final status ('success', 'error', 'cancelled')
            error: Error message if status is 'error'
        """
        pass

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve trace data.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Trace data or None if not found
        """
        pass

    @abstractmethod
    def search_traces(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search traces based on filters.
        
        Args:
            filters: Search filters
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching traces
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Flush any pending observability data to the backend.
        """
        pass


class TraceNotFoundError(Exception):
    """Raised when a requested trace is not found."""
    pass


class SpanNotFoundError(Exception):
    """Raised when a requested span is not found."""
    pass


class ObservabilityError(Exception):
    """Base exception for observability operations."""
    pass