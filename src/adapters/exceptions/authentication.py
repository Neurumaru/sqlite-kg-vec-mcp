"""
인증 및 권한 부여 관련 인프라 예외.
"""

from typing import Any, Optional

from .base import InfrastructureException


class AuthenticationException(InfrastructureException):
    """
    인증 실패.

    신원, API 키, 토큰 및 기타 인증 메커니즘을 확인하는 데
    문제가 있는 경우 사용됩니다.
    """

    def __init__(
        self,
        service: str,
        auth_type: str,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        인증 예외를 초기화합니다.

        Args:
            service: 인증이 필요한 서비스
            auth_type: 인증 유형 (API 키, 토큰 등)
            message: 상세 오류 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.service = service
        self.auth_type = auth_type

        full_message = f"{service} 인증 실패 ({auth_type}): {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "AUTHENTICATION_FAILED",
            context=context,
            original_error=original_error,
        )


class AuthorizationException(InfrastructureException):
    """
    권한 부여 실패.

    권한 거부, 불충분한 권한 및 접근 제어 위반에
    사용됩니다.
    """

    def __init__(
        self,
        service: str,
        resource: str,
        required_permission: str,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        권한 부여 예외를 초기화합니다.

        Args:
            service: 접근을 거부하는 서비스
            resource: 접근 중인 리소스
            required_permission: 접근에 필요한 권한
            message: 선택적 사용자 지정 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.service = service
        self.resource = resource
        self.required_permission = required_permission

        if message is None:
            message = f"{service}의 {resource}에 대한 접근이 거부되었습니다: '{required_permission}' 권한이 필요합니다"

        super().__init__(
            message=message,
            error_code=error_code or "AUTHORIZATION_FAILED",
            context=context,
            original_error=original_error,
        )
