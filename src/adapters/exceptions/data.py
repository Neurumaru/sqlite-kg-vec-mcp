"""
Data-related infrastructure exceptions.
"""

import re
from typing import Any

from .base import InfrastructureException


class DataException(InfrastructureException):
    """
    Base exception for data-related errors.

    This exception covers issues with data processing, validation,
    parsing, and integrity in infrastructure components.
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
        Initialize data exception.

        Args:
            operation: Operation being performed when error occurred
            message: Detailed error message
            data_type: Type of data being processed (optional)
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.operation = operation
        self.data_type = data_type

        full_message = f"Data error in {operation}: {message}"
        if data_type:
            full_message = f"Data error in {operation} ({data_type}): {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "DATA_ERROR",
            context=context,
            original_error=original_error,
        )


class DataIntegrityException(DataException):
    """
    Data integrity violations.

    Used for constraint violations, foreign key errors,
    unique constraint violations, etc.
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
        Initialize data integrity exception.

        Args:
            constraint: Name or type of constraint violated
            table: Table name where violation occurred (optional)
            message: Optional custom message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.constraint = constraint
        self.table = table

        if message is None:
            if table:
                message = f"Integrity constraint '{constraint}' violated on table '{table}'"
            else:
                message = f"Integrity constraint '{constraint}' violated"

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
    Data validation failures.

    Used for schema validation, format validation,
    and business rule validation at the data level.
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
        Initialize data validation exception.

        Args:
            field: Field name that failed validation
            value: Invalid value
            expected_format: Expected format or constraint
            message: Optional custom message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.field = field
        self.value = value
        self.expected_format = expected_format

        if message is None:
            message = f"Field '{field}' validation failed: expected {expected_format}, got {type(value).__name__}"

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
    Data parsing failures.

    Used for JSON parsing, XML parsing, CSV parsing,
    and other structured data format issues.
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
        Initialize data parsing exception.

        Args:
            data_format: Format being parsed (JSON, XML, CSV, etc.)
            message: Detailed error message
            raw_data: Raw data that failed to parse (truncated if long)
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
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
