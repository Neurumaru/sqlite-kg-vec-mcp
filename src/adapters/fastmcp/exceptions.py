"""
FastMCP 서버 인프라 예외.

이 예외들은 MCP 프로토콜 오류, 서버 생명주기 문제,
메시지 처리 실패 등을 처리합니다.
"""

from typing import Any

from ..exceptions.base import InfrastructureException
from ..exceptions.connection import ConnectionException
from ..exceptions.data import DataParsingException


class MCPException(InfrastructureException):
    """
    MCP 프로토콜 오류의 기본 예외.

    MCP 메시지 핸들링, 프로토콜 준수,
    서버 통신 관련 문제를 다룹니다.
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
        MCP 예외를 초기화합니다.

        Args:
            operation: 수행 중인 MCP 작업
            message: 상세 오류 메시지
            mcp_method: 관련된 MCP 메서드 이름
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 서버 생명주기 오류.

    서버 시작, 종료, 런타임 문제를 처리합니다.
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
        MCP 서버 예외를 초기화합니다.

        Args:
            server_state: 현재 서버 상태
            operation: 수행 중인 서버 작업
            message: 상세 오류 메시지
            port: 관련된 서버 포트
            host: 관련된 서버 호스트
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 메시지 처리 오류.

    메시지 파싱, 유효성 검사, 프로토콜 준수 관련
    문제를 처리합니다.
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
        MCP 메시지 예외를 초기화합니다.

        Args:
            message_type: MCP 메시지 유형
            message_content: 원본 메시지 내용 (일부)
            validation_error: 특정 유효성 검사 오류
            mcp_method: 관련된 MCP 메서드 이름
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 도구 실행 오류.

    도구 발견, 등록, 실행 중 발생하는 실패를 처리합니다.
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
        MCP 도구 예외를 초기화합니다.

        Args:
            tool_name: MCP 도구의 이름
            operation: 수행 중인 도구 작업
            message: 상세 오류 메시지
            tool_args: 관련된 도구 인수
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 리소스 접근 오류.

    리소스 발견, 접근 제어, 리소스 작업 관련 문제를 처리합니다.
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
        MCP 리소스 예외를 초기화합니다.

        Args:
            resource_uri: 리소스의 URI
            operation: 수행 중인 리소스 작업
            message: 상세 오류 메시지
            resource_type: 알려진 경우 리소스 유형
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 프롬프트 처리 오류.

    프롬프트 발견, 템플릿 처리, 프롬프트 실행 관련
    문제를 처리합니다.
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
        MCP 프롬프트 예외를 초기화합니다.

        Args:
            prompt_name: 프롬프트의 이름
            operation: 수행 중인 프롬프트 작업
            message: 상세 오류 메시지
            prompt_args: 관련된 프롬프트 인수
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
    MCP 클라이언트 연결 오류.

    클라이언트 연결, 전송 계층, 통신 채널 관련
    문제를 처리합니다.
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
        MCP 연결 예외를 초기화합니다.

        Args:
            transport_type: 전송 유형 (stdio, sse, websocket)
            endpoint: 연결 엔드포인트
            message: 상세 오류 메시지
            client_id: 사용 가능한 경우 클라이언트 식별자
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
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
