"""
Infrastructure exceptions for adapter layer.

This module provides common infrastructure exception base classes that can be
inherited by technology-specific exceptions in each adapter folder.
These exceptions follow hexagonal architecture principles by keeping
infrastructure concerns separate from domain logic.
"""

from .authentication import (
    AuthenticationException,
    AuthorizationException,
)
from .base import InfrastructureException
from .connection import (
    ConnectionException,
    DatabaseConnectionException,
    HTTPConnectionException,
)
from .data import (
    DataException,
    DataIntegrityException,
    DataParsingException,
    DataValidationException,
)
from .timeout import (
    DatabaseTimeoutException,
    HTTPTimeoutException,
    TimeoutException,
)

__all__ = [
    # Base
    "InfrastructureException",
    # Connection
    "ConnectionException",
    "DatabaseConnectionException",
    "HTTPConnectionException",
    # Timeout
    "TimeoutException",
    "DatabaseTimeoutException",
    "HTTPTimeoutException",
    # Data
    "DataException",
    "DataIntegrityException",
    "DataValidationException",
    "DataParsingException",
    # Authentication
    "AuthenticationException",
    "AuthorizationException",
]
