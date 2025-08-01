"""
Connection-related infrastructure exceptions.
"""

from typing import Any

from .base import InfrastructureException


class ConnectionException(InfrastructureException):
    """
    Base exception for connection-related errors.

    This exception covers failures in establishing or maintaining
    connections to external systems.
    """

    def __init__(
        self,
        service: str,
        endpoint: str,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize connection exception.

        Args:
            service: Name of the service (e.g., "SQLite", "Ollama")
            endpoint: Connection endpoint (URL, file path, etc.)
            message: Detailed error message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.service = service
        self.endpoint = endpoint

        full_message = f"{service} connection failed ({endpoint}): {message}"
        super().__init__(
            message=full_message,
            error_code=error_code or "CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )


class DatabaseConnectionException(ConnectionException):
    """
    Database connection failures.

    Used for any database connection issues including
    file access, permissions, corruption, etc.
    """

    def __init__(
        self,
        db_path: str,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize database connection exception.

        Args:
            db_path: Database file path or connection string
            message: Detailed error message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        super().__init__(
            service="Database",
            endpoint=db_path,
            message=message,
            error_code=error_code or "DB_CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )
        self.db_path = db_path


class HTTPConnectionException(ConnectionException):
    """
    HTTP connection failures.

    Used for network-related issues when connecting to
    HTTP APIs and services.
    """

    def __init__(
        self,
        url: str,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize HTTP connection exception.

        Args:
            url: Target URL
            message: Detailed error message
            status_code: HTTP status code if available
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.status_code = status_code

        if status_code:
            message = f"HTTP {status_code}: {message}"

        super().__init__(
            service="HTTP",
            endpoint=url,
            message=message,
            error_code=error_code or "HTTP_CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )
        self.url = url
