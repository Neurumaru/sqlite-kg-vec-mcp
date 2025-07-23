"""
Domain events for the knowledge graph system.
"""

from .base import DomainEvent
from .entity_created import EntityCreated
from .relationship_created import RelationshipCreated
from .search_completed import SearchCompleted

__all__ = [
    "DomainEvent",
    "EntityCreated",
    "RelationshipCreated",
    "SearchCompleted",
]
