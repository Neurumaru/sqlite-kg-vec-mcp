"""
Authentication and authorization related infrastructure exceptions.
"""

from typing import Optional, Dict, Any
from .base import InfrastructureException


class AuthenticationException(InfrastructureException):
    """
    Authentication failures.
    
    Used for issues with validating identity, API keys,
    tokens, and other authentication mechanisms.
    """

    def __init__(
        self,
        service: str,
        auth_type: str,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize authentication exception.
        
        Args:
            service: Service requiring authentication
            auth_type: Type of authentication (API key, token, etc.)
            message: Detailed error message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.service = service
        self.auth_type = auth_type
        
        full_message = f"{service} authentication failed ({auth_type}): {message}"
        
        super().__init__(
            message=full_message,
            error_code=error_code or "AUTHENTICATION_FAILED",
            context=context,
            original_error=original_error
        )


class AuthorizationException(InfrastructureException):
    """
    Authorization failures.
    
    Used for permission denied, insufficient privileges,
    and access control violations.
    """

    def __init__(
        self,
        service: str,
        resource: str,
        required_permission: str,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize authorization exception.
        
        Args:
            service: Service denying access
            resource: Resource being accessed
            required_permission: Permission required for access
            message: Optional custom message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.service = service
        self.resource = resource
        self.required_permission = required_permission
        
        if message is None:
            message = f"Access denied to {resource} on {service}: requires '{required_permission}' permission"
        
        super().__init__(
            message=message,
            error_code=error_code or "AUTHORIZATION_FAILED",
            context=context,
            original_error=original_error
        )