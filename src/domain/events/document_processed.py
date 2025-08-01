"""
문서 처리 완료 이벤트.
"""

from dataclasses import dataclass

from .base import DomainEvent


@dataclass(init=False)
class DocumentProcessed(DomainEvent):
    """
    문서 처리가 완료되었을 때 발생하는 이벤트.
    """

    document_id: str = ""
    extracted_node_count: int = 0
    extracted_relationship_count: int = 0
    processing_time_seconds: float = 0.0

    def __init__(
        self,
        event_id: str,
        occurred_at,
        event_type: str,
        aggregate_id: str,
        document_id: str,
        extracted_node_count: int,
        extracted_relationship_count: int,
        processing_time_seconds: float,
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
        self.document_id = document_id
        self.extracted_node_count = extracted_node_count
        self.extracted_relationship_count = extracted_relationship_count
        self.processing_time_seconds = processing_time_seconds

    @classmethod
    def create(  # pylint: disable=arguments-differ
        cls,
        aggregate_id: str,
        extracted_node_count: int,
        extracted_relationship_count: int,
        processing_time_seconds: float,
        **kwargs,
    ) -> "DocumentProcessed":
        """문서 처리 완료 이벤트 생성."""
        event = super().create(
            aggregate_id=aggregate_id,
            document_id=aggregate_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds,
            **kwargs,
        )
        return event
