"""
문서 관련 예외들.
"""

from .base import DomainException


class DocumentNotFoundException(DomainException):
    """문서를 찾을 수 없을 때 발생하는 예외."""

    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID '{document_id}' not found",
            error_code="DOCUMENT_NOT_FOUND",
        )
        self.document_id = document_id


class DocumentAlreadyExistsException(DomainException):
    """문서가 이미 존재할 때 발생하는 예외."""

    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID '{document_id}' already exists",
            error_code="DOCUMENT_ALREADY_EXISTS",
        )
        self.document_id = document_id


class InvalidDocumentException(DomainException):
    """유효하지 않은 문서일 때 발생하는 예외."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid document: {reason}", error_code="INVALID_DOCUMENT"
        )
        self.reason = reason


class DocumentProcessingException(DomainException):
    """문서 처리 중 오류가 발생했을 때의 예외."""

    def __init__(self, document_id: str, reason: str):
        super().__init__(
            message=f"Failed to process document '{document_id}': {reason}",
            error_code="DOCUMENT_PROCESSING_FAILED",
        )
        self.document_id = document_id
        self.reason = reason
