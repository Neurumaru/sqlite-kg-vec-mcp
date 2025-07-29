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
        super().__init__(message=f"Invalid document: {reason}", error_code="INVALID_DOCUMENT")
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


class ConcurrentModificationError(DomainException):
    """
    동시 수정 오류 예외.

    TODO: 확장성 고려 - 분산 환경에서의 동시성 제어 전략
    현재는 낙관적 잠금을 사용하지만, 분산 환경에서는 다음을 고려해야 함:
    1. 분산 락 매니저를 통한 비관적 잠금 옵션
    2. 이벤트 소싱을 통한 충돌 해결 전략
    3. CRDT(Conflict-free Replicated Data Types) 활용 검토
    """

    def __init__(self, document_id: str, expected_version: int, actual_version: int):
        super().__init__(
            message=f"Concurrent modification detected for document '{document_id}'. Expected version {expected_version}, but was {actual_version}",
            error_code="CONCURRENT_MODIFICATION_ERROR",
        )
        self.document_id = document_id
        self.expected_version = expected_version
        self.actual_version = actual_version
