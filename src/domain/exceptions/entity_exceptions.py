"""
Entity-related domain exceptions.
"""

from .base import DomainException


class EntityNotFoundException(DomainException):
    """Raised when an entity is not found."""

    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity with ID '{entity_id}' not found",
            error_code="ENTITY_NOT_FOUND"
        )
        self.entity_id = entity_id


class EntityAlreadyExistsException(DomainException):
    """Raised when trying to create an entity that already exists."""

    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity with ID '{entity_id}' already exists",
            error_code="ENTITY_ALREADY_EXISTS"
        )
        self.entity_id = entity_id


class InvalidEntityException(DomainException):
    """Raised when entity data is invalid."""

    def __init__(self, message: str):
        super().__init__(
            f"Invalid entity: {message}",
            error_code="INVALID_ENTITY"
        )
