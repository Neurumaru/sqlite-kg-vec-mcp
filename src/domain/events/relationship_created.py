"""
관계 생성 이벤트.
"""

from dataclasses import dataclass

from .base import DomainEvent


@dataclass(init=False)
class RelationshipCreated(DomainEvent):
    """
    새로운 관계가 생성되었을 때 발생하는 이벤트.
    """

    relationship_id: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    relationship_type: str = ""
    relationship_label: str = ""
    confidence: float = 0.0
    source_document_id: str | None = None

    def __init__(
        self,
        event_id: str,
        occurred_at,
        event_type: str,
        aggregate_id: str,
        relationship_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        relationship_label: str,
        confidence: float,
        source_document_id: str | None = None,
        version: int = 1,
        metadata=None,
    ):
        super().__init__(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type=event_type,
            aggregate_id=aggregate_id,
            version=version,
            metadata=metadata or {},
        )
        self.relationship_id = relationship_id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.relationship_type = relationship_type
        self.relationship_label = relationship_label
        self.confidence = confidence
        self.source_document_id = source_document_id

    @classmethod
    def create(  # pylint: disable=arguments-differ
        cls,
        aggregate_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        relationship_label: str,
        confidence: float,
        source_document_id: str | None = None,
        **kwargs,
    ) -> "RelationshipCreated":
        """관계 생성 이벤트 생성."""
        event = super().create(
            aggregate_id=aggregate_id,
            relationship_id=aggregate_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            relationship_label=relationship_label,
            confidence=confidence,
            source_document_id=source_document_id,
            **kwargs,
        )
        return event
