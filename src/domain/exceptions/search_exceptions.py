"""
Search-related domain exceptions.
"""

from .base import DomainException


class InvalidSearchCriteriaException(DomainException):
    """Raised when search criteria is invalid."""

    def __init__(self, message: str):
        super().__init__(
            f"Invalid search criteria: {message}",
            error_code="INVALID_SEARCH_CRITERIA"
        )


class SearchFailedException(DomainException):
    """Raised when a search operation fails."""

    def __init__(self, message: str):
        super().__init__(
            f"Search failed: {message}",
            error_code="SEARCH_FAILED"
        )
