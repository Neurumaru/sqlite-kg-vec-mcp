"""
Domain layer for the SQLite Knowledge Graph Vector MCP system.

This package contains the core business logic and domain models
following Domain-Driven Design principles.
"""

# Value Objects
from .value_objects import (
    NodeId,
    Vector,
    SearchCriteria,
    EntityType,
    RelationshipType,
)

# Entities
from .entities import (
    Entity,
    Relationship,
    Embedding,
    SearchResult,
    SearchResultCollection,
    KnowledgeGraph,
)

# Events
from .events import (
    DomainEvent,
    EntityCreated,
    RelationshipCreated,
    SearchCompleted,
)

# Exceptions
from .exceptions import (
    DomainException,
    EntityNotFoundException,
    EntityAlreadyExistsException,
    InvalidEntityException,
    RelationshipNotFoundException,
    InvalidRelationshipException,
    InvalidSearchCriteriaException,
    SearchFailedException,
)

__all__ = [
    # Value Objects
    "NodeId",
    "Vector",
    "SearchCriteria",
    "EntityType",
    "RelationshipType",

    # Entities
    "Entity",
    "Relationship",
    "Embedding",
    "SearchResult",
    "SearchResultCollection",
    "KnowledgeGraph",

    # Events
    "DomainEvent",
    "EntityCreated",
    "RelationshipCreated",
    "SearchCompleted",

    # Exceptions
    "DomainException",
    "EntityNotFoundException",
    "EntityAlreadyExistsException",
    "InvalidEntityException",
    "RelationshipNotFoundException",
    "InvalidRelationshipException",
    "InvalidSearchCriteriaException",
    "SearchFailedException",
]
