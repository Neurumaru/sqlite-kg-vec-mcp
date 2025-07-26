"""
Base infrastructure exception classes.
"""

from typing import Optional, Dict, Any


class InfrastructureException(Exception):
    """
    Base exception for all infrastructure-related errors.
    
    This exception represents technical failures in external systems,
    databases, APIs, and other infrastructure components. It serves
    as the root of the infrastructure exception hierarchy.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize infrastructure exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for categorization
            context: Additional context information
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the exception."""
        self.context[key] = value

    def get_context(self) -> Dict[str, Any]:
        """Get all context information."""
        return self.context.copy()