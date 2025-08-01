"""
관계 관련 예외들.
"""

from .base import DomainException


class RelationshipNotFoundException(DomainException):
    """관계를 찾을 수 없을 때 발생하는 예외."""

    def __init__(self, relationship_id: str):
        super().__init__(
            message=f"Relationship with ID '{relationship_id}' not found",
            error_code="RELATIONSHIP_NOT_FOUND",
        )
        self.relationship_id = relationship_id


class RelationshipAlreadyExistsException(DomainException):
    """관계가 이미 존재할 때 발생하는 예외."""

    def __init__(self, relationship_id: str):
        super().__init__(
            message=f"Relationship with ID '{relationship_id}' already exists",
            error_code="RELATIONSHIP_ALREADY_EXISTS",
        )
        self.relationship_id = relationship_id


class InvalidRelationshipException(DomainException):
    """유효하지 않은 관계일 때 발생하는 예외."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid relationship: {reason}", error_code="INVALID_RELATIONSHIP"
        )
        self.reason = reason


class RelationshipEmbeddingException(DomainException):
    """관계 임베딩 관련 오류."""

    def __init__(self, relationship_id: str, reason: str):
        super().__init__(
            message=f"Relationship embedding error for '{relationship_id}': {reason}",
            error_code="RELATIONSHIP_EMBEDDING_ERROR",
        )
        self.relationship_id = relationship_id
        self.reason = reason


class CircularRelationshipException(DomainException):
    """순환 관계 오류."""

    def __init__(self, node_id: str):
        super().__init__(
            message=f"Circular relationship detected for node '{node_id}'",
            error_code="CIRCULAR_RELATIONSHIP",
        )
        self.node_id = node_id
