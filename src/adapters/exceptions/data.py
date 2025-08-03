"""
데이터 관련 인프라 예외.
"""

import re
from typing import Any

from .base import InfrastructureException


class DataException(InfrastructureException):
    """
    데이터 관련 오류의 기본 예외.

    이 예외는 인프라 구성 요소의 데이터 처리, 유효성 검사,
    파싱 및 무결성 문제를 다룹니다.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        data_type: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        데이터 예외를 초기화합니다.

        Args:
            operation: 오류 발생 시 수행 중이던 작업
            message: 상세 오류 메시지
            data_type: 처리 중인 데이터 유형 (선택 사항)
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.operation = operation
        self.data_type = data_type

        full_message = f"{operation}에서 데이터 오류 발생: {message}"
        if data_type:
            full_message = f"{operation} ({data_type})에서 데이터 오류 발생: {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "DATA_ERROR",
            context=context,
            original_error=original_error,
        )


class DataIntegrityException(DataException):
    """
    데이터 무결성 위반.

    제약 조건 위반, 외래 키 오류,
    고유 제약 조건 위반 등에 사용됩니다.
    """

    def __init__(
        self,
        constraint: str,
        table: str | None = None,
        message: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        데이터 무결성 예외를 초기화합니다.

        Args:
            constraint: 위반된 제약 조건의 이름 또는 유형
            table: 위반이 발생한 테이블 이름 (선택 사항)
            message: 선택적 사용자 지정 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.constraint = constraint
        self.table = table

        if message is None:
            if table:
                message = f"테이블 '{table}'에서 무결성 제약 조건 '{constraint}' 위반"
            else:
                message = f"무결성 제약 조건 '{constraint}' 위반"

        super().__init__(
            operation="data integrity check",
            message=message,
            data_type="constraint",
            error_code=error_code or "DATA_INTEGRITY_VIOLATION",
            context=context,
            original_error=original_error,
        )


class DataValidationException(DataException):
    """
    데이터 유효성 검사 실패.

    스키마 유효성 검사, 형식 유효성 검사 및
    데이터 수준의 비즈니스 규칙 유효성 검사에 사용됩니다.
    """

    def __init__(
        self,
        field: str,
        value: str | int | float | bool | None,
        expected_format: str,
        message: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        데이터 유효성 검사 예외를 초기화합니다.

        Args:
            field: 유효성 검사에 실패한 필드 이름
            value: 잘못된 값
            expected_format: 예상 형식 또는 제약 조건
            message: 선택적 사용자 지정 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.field = field
        self.value = value
        self.expected_format = expected_format

        if message is None:
            message = f"필드 '{field}' 유효성 검사 실패: 예상 형식 {expected_format}, 실제 형식 {type(value).__name__}"

        super().__init__(
            operation="data validation",
            message=message,
            data_type="validation",
            error_code=error_code or "DATA_VALIDATION_FAILED",
            context=context,
            original_error=original_error,
        )


class DataParsingException(DataException):
    """
    데이터 파싱 실패.

    JSON 파싱, XML 파싱, CSV 파싱 및
    기타 구조화된 데이터 형식 문제에 사용됩니다.
    """

    def __init__(
        self,
        data_format: str,
        message: str,
        raw_data: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        데이터 파싱 예외를 초기화합니다.

        Args:
            data_format: 파싱 중인 형식 (JSON, XML, CSV 등)
            message: 상세 오류 메시지
            raw_data: 파싱에 실패한 원시 데이터 (길면 잘림)
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.data_format = data_format
        self.raw_data = raw_data

        if raw_data and len(raw_data) > 200:
            # 민감한 패턴 마스킹 (이메일, API 키 등)
            masked_data = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", raw_data[:200]
            )
            masked_data = re.sub(r"\b[A-Za-z0-9]{32,}\b", "[API_KEY]", masked_data)
            self.raw_data = masked_data + "..."

        super().__init__(
            operation=f"{data_format} parsing",
            message=message,
            data_type=data_format,
            error_code=error_code or "DATA_PARSING_FAILED",
            context=context,
            original_error=original_error,
        )
