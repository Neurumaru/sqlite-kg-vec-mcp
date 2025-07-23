"""
Relationship-related domain exceptions.
"""

from .base import DomainException


class RelationshipNotFoundException(DomainException):
    """Raised when a relationship is not found."""

    def __init__(self, relationship_id: str):
        super().__init__(
            f"Relationship with ID '{relationship_id}' not found",
            error_code="RELATIONSHIP_NOT_FOUND"
        )
        self.relationship_id = relationship_id


class InvalidRelationshipException(DomainException):
    """Raised when relationship data is invalid."""

    def __init__(self, message: str):
        super().__init__(
            f"Invalid relationship: {message}",
            error_code="INVALID_RELATIONSHIP"
        )
