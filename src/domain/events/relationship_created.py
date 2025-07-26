"""
Relationship creation domain event.
"""

from dataclasses import dataclass

from src.domain.value_objects.node_id import NodeId
from .base import DomainEvent


@dataclass(frozen=True)
class RelationshipCreated(DomainEvent):
    """Domain event raised when a new relationship is created."""

    relationship_id: NodeId
    source_id: NodeId
    target_id: NodeId
    relationship_type: str
    graph_id: NodeId

    @property
    def event_type(self) -> str:
        return "relationship_created"
