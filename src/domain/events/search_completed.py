"""
Search completion domain event.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from ..value_objects.node_id import NodeId
from .base import DomainEvent


@dataclass(frozen=True)
class SearchCompleted(DomainEvent):
    """Domain event raised when a search operation is completed."""

    search_id: NodeId
    query_text: Optional[str]
    results_count: int
    total_time_ms: float
    search_metadata: Optional[Dict[str, Any]] = None

    @property
    def event_type(self) -> str:
        return "search_completed"
