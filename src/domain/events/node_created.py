"""
노드 생성 이벤트.
"""

from dataclasses import dataclass

from .base import DomainEvent


@dataclass(init=False)
class NodeCreated(DomainEvent):
    """
    새로운 노드가 생성되었을 때 발생하는 이벤트.
    """

    node_id: str = ""
    node_name: str = ""
    node_type: str = ""
    source_document_id: str | None = None

    def __init__(
        self,
        event_id: str,
        occurred_at,
        event_type: str,
        aggregate_id: str,
        node_id: str,
        node_name: str,
        node_type: str,
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
        self.node_id = node_id
        self.node_name = node_name
        self.node_type = node_type
        self.source_document_id = source_document_id

    @classmethod
    def create(  # pylint: disable=arguments-differ
        cls,
        aggregate_id: str,
        node_name: str,
        node_type: str,
        source_document_id: str | None = None,
        **kwargs,
    ) -> "NodeCreated":
        """노드 생성 이벤트 생성."""
        event = super().create(
            aggregate_id=aggregate_id,
            node_id=aggregate_id,
            node_name=node_name,
            node_type=node_type,
            source_document_id=source_document_id,
            **kwargs,
        )
        return event
