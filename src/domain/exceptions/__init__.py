"""
Domain exceptions for the knowledge graph system.
"""

from .base import DomainException
from .entity_exceptions import (
    EntityNotFoundException,
    EntityAlreadyExistsException,
    InvalidEntityException,
)
from .relationship_exceptions import (
    RelationshipNotFoundException,
    InvalidRelationshipException,
)
from .search_exceptions import (
    InvalidSearchCriteriaException,
    SearchFailedException,
)

__all__ = [
    "DomainException",
    "EntityNotFoundException",
    "EntityAlreadyExistsException",
    "InvalidEntityException",
    "RelationshipNotFoundException",
    "InvalidRelationshipException",
    "InvalidSearchCriteriaException",
    "SearchFailedException",
]
