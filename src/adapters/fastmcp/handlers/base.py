"""
Base handler for MCP operations.
"""

import logging
from abc import ABC
from typing import Any

from ..config import FastMCPConfig


class BaseHandler(ABC):
    """
    Base class for MCP operation handlers.

    Provides common functionality like logging and response formatting.
    """

    def __init__(self, config: FastMCPConfig):
        """
        Initialize base handler.

        Args:
            config: FastMCP configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.log_level))

    def _create_success_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a standardized success response."""
        return {"success": True, **data}

    def _create_error_response(self, message: str, error_code: str | None = None) -> dict[str, Any]:
        """Create a standardized error response."""
        return {
            "success": False,
            "error": message,
            "error_code": error_code,
        }
