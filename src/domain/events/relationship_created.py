"""
관계 생성 이벤트.
"""

from dataclasses import dataclass
from typing import Optional

from .base import DomainEvent


@dataclass
class RelationshipCreated(DomainEvent):
    """
    새로운 관계가 생성되었을 때 발생하는 이벤트.
    """

    relationship_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    relationship_label: str
    confidence: float
    source_document_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        relationship_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        relationship_label: str,
        confidence: float,
        source_document_id: Optional[str] = None,
    ) -> "RelationshipCreated":
        """관계 생성 이벤트 생성."""
        return super().create(
            aggregate_id=relationship_id,
            relationship_id=relationship_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            relationship_label=relationship_label,
            confidence=confidence,
            source_document_id=source_document_id,
        )
