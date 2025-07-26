"""
Domain layer for the SQLite Knowledge Graph Vector MCP system.

This package contains the core business logic and domain models
following Domain-Driven Design principles.
"""

# Value Objects - direct imports to avoid circular dependencies during testing
from .value_objects.node_id import NodeId
from .value_objects.vector import Vector
from .value_objects.search_criteria import SearchCriteria
from .value_objects.entity_type import EntityType
from .value_objects.relationship_type import RelationshipType

# Temporarily comment out complex imports for testing
# from .entities import (
#     Entity,
#     Relationship,
#     Embedding,
#     SearchResult,
#     SearchResultCollection,
#     KnowledgeGraph,
# )

# from .events import (
#     DomainEvent,
#     EntityCreated,
#     RelationshipCreated,
#     SearchCompleted,
# )

# from .exceptions import (
#     DomainException,
#     EntityNotFoundException,
#     EntityAlreadyExistsException,
#     InvalidEntityException,
#     RelationshipNotFoundException,
#     InvalidRelationshipException,
#     InvalidSearchCriteriaException,
#     SearchFailedException,
# )

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
