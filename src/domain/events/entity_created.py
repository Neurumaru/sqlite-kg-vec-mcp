"""
Entity creation domain event.
"""

from dataclasses import dataclass
from typing import Optional

from ..value_objects.node_id import NodeId
from .base import DomainEvent


@dataclass(frozen=True)
class EntityCreated(DomainEvent):
    """Domain event raised when a new entity is created."""

    entity_id: NodeId
    entity_type: str
    name: Optional[str]
    graph_id: NodeId

    @property
    def event_type(self) -> str:
        return "entity_created"
