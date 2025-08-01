"""
Timeout-related infrastructure exceptions.
"""

from typing import Any

from .base import InfrastructureException


class TimeoutException(InfrastructureException):
    """
    Base exception for timeout-related errors.

    This exception covers timeouts in operations with external systems.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        message: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize timeout exception.

        Args:
            operation: Description of the operation that timed out
            timeout_duration: Timeout duration in seconds
            message: Optional custom message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.operation = operation
        self.timeout_duration = timeout_duration

        if message is None:
            message = f"Operation '{operation}' timed out after {timeout_duration}s"

        super().__init__(
            message=message,
            error_code=error_code or "OPERATION_TIMEOUT",
            context=context,
            original_error=original_error,
        )


class DatabaseTimeoutException(TimeoutException):
    """
    Database operation timeouts.

    Used for database queries, transactions, or connection
    operations that exceed time limits.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        query: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize database timeout exception.

        Args:
            operation: Database operation description
            timeout_duration: Timeout duration in seconds
            query: SQL query that timed out (optional)
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.query = query

        message = f"Database {operation} timed out after {timeout_duration}s"
        if query:
            message += f" (Query: {query[:100]}...)"

        super().__init__(
            operation=f"Database {operation}",
            timeout_duration=timeout_duration,
            message=message,
            error_code=error_code or "DB_TIMEOUT",
            context=context,
            original_error=original_error,
        )


class HTTPTimeoutException(TimeoutException):
    """
    HTTP request timeouts.

    Used for HTTP API calls that exceed time limits.
    """

    def __init__(
        self,
        url: str,
        method: str,
        timeout_duration: float,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize HTTP timeout exception.

        Args:
            url: Target URL
            method: HTTP method (GET, POST, etc.)
            timeout_duration: Timeout duration in seconds
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.url = url
        self.method = method

        message = f"HTTP {method} request to {url} timed out after {timeout_duration}s"

        super().__init__(
            operation=f"HTTP {method}",
            timeout_duration=timeout_duration,
            message=message,
            error_code=error_code or "HTTP_TIMEOUT",
            context=context,
            original_error=original_error,
        )
