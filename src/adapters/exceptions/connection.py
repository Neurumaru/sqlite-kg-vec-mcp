"""
연결 관련 인프라 예외.
"""

from typing import Any

from .base import InfrastructureException


class ConnectionException(InfrastructureException):
    """
    연결 관련 오류의 기본 예외.

    이 예외는 외부 시스템과의 연결 설정 또는 유지 실패를 다룹니다.
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
        연결 예외를 초기화합니다.

        Args:
            service: 서비스 이름 (예: "SQLite", "Ollama")
            endpoint: 연결 엔드포인트 (URL, 파일 경로 등)
            message: 상세 오류 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.service = service
        self.endpoint = endpoint

        full_message = f"{service} 연결 실패 ({endpoint}): {message}"
        super().__init__(
            message=full_message,
            error_code=error_code or "CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )


class DatabaseConnectionException(ConnectionException):
    """
    데이터베이스 연결 실패.

    파일 접근, 권한, 손상 등을 포함한 모든 데이터베이스 연결 문제에
    사용됩니다.
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
        데이터베이스 연결 예외를 초기화합니다.

        Args:
            db_path: 데이터베이스 파일 경로 또는 연결 문자열
            message: 상세 오류 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
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
    HTTP 연결 실패.

    HTTP API 및 서비스에 연결할 때 발생하는 네트워크 관련 문제에
    사용됩니다.
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
        HTTP 연결 예외를 초기화합니다.

        Args:
            url: 대상 URL
            message: 상세 오류 메시지
            status_code: 사용 가능한 경우 HTTP 상태 코드
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
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
