"""
Entity creation domain event.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.domain.value_objects.node_id import NodeId
from src.domain.events.base import DomainEvent


@dataclass(frozen=True)
class EntityCreated(DomainEvent):
    """Domain event raised when a new entity is created."""

    entity_id: NodeId
    entity_type: str
    graph_id: NodeId
    name: Optional[str] = None

    @property
    def event_type(self) -> str:
        return "entity_created"
