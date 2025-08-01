"""
Unit tests for FastMCP adapter exceptions.
"""

import unittest

# Import FastMCP-specific exceptions
from src.adapters.fastmcp.exceptions import (
    MCPException,
    MCPServerException,
    MCPToolException,
)


class TestMCPException(unittest.TestCase):
    """Test cases for MCPException class."""

    def test_init_with_required_parameters(self):
        """
        Given: 필수 매개변수만으로 MCPException 생성
        When: exception을 초기화할 때
        Then: 모든 속성이 올바르게 설정되어야 함
        """
        # Given
        operation = "test_operation"
        message = "Test error message"

        # When
        exception = MCPException(operation=operation, message=message)

        # Then
        self.assertEqual(exception.operation, operation)
        self.assertIn(message, str(exception))
        self.assertIn(operation, str(exception))
        self.assertEqual(exception.error_code, "MCP_ERROR")

    def test_init_with_all_parameters(self):
        """
        Given: 모든 매개변수로 MCPException 생성
        When: exception을 초기화할 때
        Then: 모든 속성이 올바르게 설정되어야 함
        """
        # Given
        operation = "test_operation"
        message = "Test error message"
        mcp_method = "test_method"
        error_code = "TEST_ERROR"
        context = {"key": "value"}
        original_error = ValueError("Original error")

        # When
        exception = MCPException(
            operation=operation,
            message=message,
            mcp_method=mcp_method,
            error_code=error_code,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.operation, operation)
        self.assertEqual(exception.mcp_method, mcp_method)
        self.assertEqual(exception.error_code, error_code)
        self.assertEqual(exception.context, context)
        self.assertEqual(exception.original_error, original_error)
        self.assertIn(mcp_method, str(exception))

    def test_message_formatting(self):
        """
        Given: MCP 메서드명이 있는 MCPException
        When: 문자열로 변환할 때
        Then: 메시지에 메서드명이 포함되어야 함
        """
        # Given
        operation = "test_operation"
        message = "Test error"
        mcp_method = "test_method"

        # When
        exception = MCPException(operation=operation, message=message, mcp_method=mcp_method)

        # Then
        expected_message = f"[MCP_ERROR] MCP {operation} failed: {message} (method: {mcp_method})"
        self.assertEqual(str(exception), expected_message)


class TestMCPServerException(unittest.TestCase):
    """Test cases for MCPServerException class."""

    def test_init_with_required_parameters(self):
        """
        Given: 필수 매개변수로 MCPServerException 생성
        When: exception을 초기화할 때
        Then: 모든 속성이 올바르게 설정되어야 함
        """
        # Given
        server_state = "starting"
        operation = "startup"
        message = "Server startup failed"

        # When
        exception = MCPServerException(
            server_state=server_state, operation=operation, message=message
        )

        # Then
        self.assertEqual(exception.server_state, server_state)
        self.assertEqual(exception.port, None)
        self.assertEqual(exception.host, None)
        self.assertIn(server_state, str(exception))
        self.assertIn(operation, str(exception))
        self.assertIn(message, str(exception))

    def test_init_with_host_and_port(self):
        """
        Given: host와 port가 포함된 MCPServerException 생성
        When: exception을 초기화할 때
        Then: 메시지에 서버 정보가 포함되어야 함
        """
        # Given
        server_state = "running"
        operation = "request_handling"
        message = "Request handling failed"
        host = "localhost"
        port = 8080

        # When
        exception = MCPServerException(
            server_state=server_state,
            operation=operation,
            message=message,
            host=host,
            port=port,
        )

        # Then
        self.assertEqual(exception.host, host)
        self.assertEqual(exception.port, port)
        self.assertIn(f"{host}:{port}", str(exception))

    def test_inheritance(self):
        """
        Given: MCPServerException 인스턴스
        When: 상속 관계를 확인할 때
        Then: MCPException을 상속해야 함
        """
        # Given & When
        exception = MCPServerException(server_state="test", operation="test", message="test")

        # Then
        self.assertIsInstance(exception, MCPException)


class TestMCPToolException(unittest.TestCase):
    """Test cases for MCPToolException class."""

    def test_init_with_required_parameters(self):
        """
        Given: 필수 매개변수로 MCPToolException 생성
        When: exception을 초기화할 때
        Then: 모든 속성이 올바르게 설정되어야 함
        """
        # Given
        tool_name = "test_tool"
        operation = "execution"
        message = "Tool execution failed"

        # When
        exception = MCPToolException(tool_name=tool_name, operation=operation, message=message)

        # Then
        self.assertEqual(exception.tool_name, tool_name)
        self.assertEqual(exception.tool_args, {})
        self.assertIn(tool_name, str(exception))
        self.assertIn(operation, str(exception))

    def test_init_with_tool_args(self):
        """
        Given: 도구 인수가 포함된 MCPToolException 생성
        When: exception을 초기화할 때
        Then: 도구 인수가 올바르게 저장되어야 함
        """
        # Given
        tool_name = "test_tool"
        operation = "execution"
        message = "Tool execution failed"
        tool_args = {"param1": "value1", "param2": 42}

        # When
        exception = MCPToolException(
            tool_name=tool_name,
            operation=operation,
            message=message,
            tool_args=tool_args,
        )

        # Then
        self.assertEqual(exception.tool_args, tool_args)


if __name__ == "__main__":
    unittest.main()
