"""
FastMCP server infrastructure exceptions.

These exceptions handle MCP protocol errors, server lifecycle issues,
and message processing failures.
"""

from typing import Any

from ..exceptions.base import InfrastructureException
from ..exceptions.connection import ConnectionException
from ..exceptions.data import DataParsingException


class MCPException(InfrastructureException):
    """
    Base exception for MCP protocol errors.

    Covers issues with MCP message handling, protocol compliance,
    and server communication.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        mcp_method: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP exception.

        Args:
            operation: MCP operation being performed
            message: Detailed error message
            mcp_method: MCP method name if relevant
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.operation = operation
        self.mcp_method = mcp_method

        full_message = f"MCP {operation} failed: {message}"
        if mcp_method:
            full_message += f" (method: {mcp_method})"

        super().__init__(
            message=full_message,
            error_code=error_code or "MCP_ERROR",
            context=context,
            original_error=original_error,
        )


class MCPServerException(MCPException):
    """
    MCP server lifecycle errors.

    Handles server startup, shutdown, and runtime issues.
    """

    def __init__(
        self,
        server_state: str,
        operation: str,
        message: str,
        port: int | None = None,
        host: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP server exception.

        Args:
            server_state: Current server state
            operation: Server operation being performed
            message: Detailed error message
            port: Server port if relevant
            host: Server host if relevant
            context: Additional context
            original_error: Original exception
        """
        self.server_state = server_state
        self.port = port
        self.host = host

        full_message = f"MCP server {operation} failed in state '{server_state}': {message}"
        if host and port:
            full_message += f" (server: {host}:{port})"

        super().__init__(
            operation=f"server {operation}",
            message=full_message,
            error_code="MCP_SERVER_ERROR",
            context=context,
            original_error=original_error,
        )


class MCPMessageException(DataParsingException):
    """
    MCP message processing errors.

    Handles issues with message parsing, validation,
    and protocol compliance.
    """

    def __init__(
        self,
        message_type: str,
        message_content: str,
        validation_error: str,
        mcp_method: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP message exception.

        Args:
            message_type: Type of MCP message
            message_content: Raw message content (truncated)
            validation_error: Specific validation error
            mcp_method: MCP method name if available
            context: Additional context
            original_error: Original exception
        """
        self.message_type = message_type
        self.mcp_method = mcp_method

        message = f"MCP {message_type} message validation failed: {validation_error}"
        if mcp_method:
            message += f" (method: {mcp_method})"

        super().__init__(
            data_format=f"MCP {message_type}",
            message=message,
            raw_data=message_content,
            error_code="MCP_MESSAGE_INVALID",
            context=context,
            original_error=original_error,
        )


class MCPToolException(MCPException):
    """
    MCP tool execution errors.

    Handles failures during tool discovery, registration,
    and execution.
    """

    def __init__(
        self,
        tool_name: str,
        operation: str,
        message: str,
        tool_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP tool exception.

        Args:
            tool_name: Name of the MCP tool
            operation: Tool operation being performed
            message: Detailed error message
            tool_args: Tool arguments if relevant
            context: Additional context
            original_error: Original exception
        """
        self.tool_name = tool_name
        self.tool_args = tool_args or {}

        full_message = f"MCP tool '{tool_name}' {operation} failed: {message}"

        super().__init__(
            operation=f"tool {operation}",
            message=full_message,
            error_code="MCP_TOOL_ERROR",
            context=context,
            original_error=original_error,
        )


class MCPResourceException(MCPException):
    """
    MCP resource access errors.

    Handles issues with resource discovery, access control,
    and resource operations.
    """

    def __init__(
        self,
        resource_uri: str,
        operation: str,
        message: str,
        resource_type: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP resource exception.

        Args:
            resource_uri: URI of the resource
            operation: Resource operation being performed
            message: Detailed error message
            resource_type: Type of resource if known
            context: Additional context
            original_error: Original exception
        """
        self.resource_uri = resource_uri
        self.resource_type = resource_type

        full_message = f"MCP resource {operation} failed for '{resource_uri}': {message}"
        if resource_type:
            full_message += f" (type: {resource_type})"

        super().__init__(
            operation=f"resource {operation}",
            message=full_message,
            error_code="MCP_RESOURCE_ERROR",
            context=context,
            original_error=original_error,
        )


class MCPPromptException(MCPException):
    """
    MCP prompt handling errors.

    Handles issues with prompt discovery, template processing,
    and prompt execution.
    """

    def __init__(
        self,
        prompt_name: str,
        operation: str,
        message: str,
        prompt_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP prompt exception.

        Args:
            prompt_name: Name of the prompt
            operation: Prompt operation being performed
            message: Detailed error message
            prompt_args: Prompt arguments if relevant
            context: Additional context
            original_error: Original exception
        """
        self.prompt_name = prompt_name
        self.prompt_args = prompt_args or {}

        full_message = f"MCP prompt '{prompt_name}' {operation} failed: {message}"

        super().__init__(
            operation=f"prompt {operation}",
            message=full_message,
            error_code="MCP_PROMPT_ERROR",
            context=context,
            original_error=original_error,
        )


class MCPConnectionException(ConnectionException):
    """
    MCP client connection errors.

    Handles issues with client connections, transport layers,
    and communication channels.
    """

    def __init__(
        self,
        transport_type: str,
        endpoint: str,
        message: str,
        client_id: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize MCP connection exception.

        Args:
            transport_type: Type of transport (stdio, sse, websocket)
            endpoint: Connection endpoint
            message: Detailed error message
            client_id: Client identifier if available
            context: Additional context
            original_error: Original exception
        """
        self.transport_type = transport_type
        self.client_id = client_id

        full_message = f"MCP connection via {transport_type}: {message}"
        if client_id:
            full_message += f" (client: {client_id})"

        super().__init__(
            service="MCP",
            endpoint=endpoint,
            message=full_message,
            error_code="MCP_CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )
