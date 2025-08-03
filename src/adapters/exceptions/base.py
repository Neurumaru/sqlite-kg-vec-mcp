"""
기본 인프라 예외 클래스.
"""

from typing import Any


class InfrastructureException(Exception):
    """
    모든 인프라 관련 오류를 위한 기본 예외.

    이 예외는 외부 시스템, 데이터베이스, API 및 기타 인프라 컴포넌트의
    기술적 실패를 나타냅니다. 이는 인프라 예외 계층의 루트 역할을 합니다.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        인프라 예외를 초기화합니다.

        인자:
            message: 사람이 읽을 수 있는 오류 메시지
            error_code: 분류를 위한 선택적 오류 코드
            context: 추가 컨텍스트 정보
            original_error: 이 오류를 발생시킨 원본 예외
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """예외의 문자열 표현을 반환합니다."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def add_context(self, key: str, value: Any) -> None:
        """예외에 컨텍스트 정보를 추가합니다."""
        self.context[key] = value

    def get_context(self) -> dict[str, Any]:
        """모든 컨텍스트 정보를 가져옵니다."""
        return self.context.copy()
