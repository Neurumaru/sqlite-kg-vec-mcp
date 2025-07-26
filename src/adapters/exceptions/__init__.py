"""
Infrastructure exceptions for adapter layer.

This module provides common infrastructure exception base classes that can be
inherited by technology-specific exceptions in each adapter folder.
These exceptions follow hexagonal architecture principles by keeping
infrastructure concerns separate from domain logic.
"""

from .base import InfrastructureException
from .connection import (
    ConnectionException,
    DatabaseConnectionException,
    HTTPConnectionException,
)
from .timeout import (
    TimeoutException,
    DatabaseTimeoutException,
    HTTPTimeoutException,
)
from .data import (
    DataException,
    DataIntegrityException,
    DataValidationException,
    DataParsingException,
)
from .authentication import (
    AuthenticationException,
    AuthorizationException,
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