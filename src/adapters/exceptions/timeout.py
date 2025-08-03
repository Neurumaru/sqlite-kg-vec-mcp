"""
타임아웃 관련 인프라 예외.
"""

from typing import Any

from .base import InfrastructureException


class TimeoutException(InfrastructureException):
    """
    타임아웃 관련 오류의 기본 예외.

    이 예외는 외부 시스템과의 작업에서 발생하는 타임아웃을 다룹니다.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        타임아웃 예외를 초기화합니다.

        Args:
            operation: 타임아웃된 작업에 대한 설명
            timeout_duration: 타임아웃 기간(초)
            message: 선택적 사용자 지정 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.operation = operation
        self.timeout_duration = timeout_duration

        if message is None:
            message = f"'{operation}' 작업이 {timeout_duration}초 후 타임아웃되었습니다"

        super().__init__(
            message=message,
            error_code=error_code or "OPERATION_TIMEOUT",
            context=context,
            original_error=original_error,
        )


class DatabaseTimeoutException(TimeoutException):
    """
    데이터베이스 작업 타임아웃.

    시간 제한을 초과하는 데이터베이스 쿼리, 트랜잭션 또는 연결 작업에
    사용됩니다.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        query: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        데이터베이스 타임아웃 예외를 초기화합니다.

        Args:
            operation: 데이터베이스 작업 설명
            timeout_duration: 타임아웃 기간(초)
            query: 타임아웃된 SQL 쿼리 (선택 사항)
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.query = query

        message = f"데이터베이스 {operation}이(가) {timeout_duration}초 후 타임아웃되었습니다"
        if query:
            message += f" (쿼리: {query[:100]}...)"

        super().__init__(
            operation=f"데이터베이스 {operation}",
            timeout_duration=timeout_duration,
            message=message,
            error_code=error_code or "DB_TIMEOUT",
            context=context,
            original_error=original_error,
        )


class HTTPTimeoutException(TimeoutException):
    """
    HTTP 요청 타임아웃.

    시간 제한을 초과하는 HTTP API 호출에 사용됩니다.
    """

    def __init__(
        self,
        url: str,
        method: str,
        timeout_duration: float,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        HTTP 타임아웃 예외를 초기화합니다.

        Args:
            url: 대상 URL
            method: HTTP 메서드 (GET, POST 등)
            timeout_duration: 타임아웃 기간(초)
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.url = url
        self.method = method

        message = f"{url}에 대한 HTTP {method} 요청이 {timeout_duration}초 후 타임아웃되었습니다"

        super().__init__(
            operation=f"HTTP {method}",
            timeout_duration=timeout_duration,
            message=message,
            error_code=error_code or "HTTP_TIMEOUT",
            context=context,
            original_error=original_error,
        )
