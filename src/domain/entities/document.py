"""
문서 도메인 엔티티.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class DocumentStatus(Enum):
    """문서 처리 상태."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentType(Enum):
    """문서 타입."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class Document:
    """
    문서 도메인 엔티티.

    지식 그래프의 기반이 되는 문서를 나타냅니다.
    문서로부터 노드와 관계가 추출되어 연결됩니다.
    """

    id: DocumentId
    title: str
    content: str
    doc_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None

    # 연결된 노드와 관계들
    connected_nodes: list[NodeId] = field(default_factory=list)
    connected_relationships: list[RelationshipId] = field(default_factory=list)

    def __post_init__(self):
        if not self.title.strip():
            raise ValueError("Document title cannot be empty")
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")

    def mark_as_processing(self) -> None:
        """문서를 처리 중 상태로 변경."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now()

    def mark_as_processed(self) -> None:
        """문서를 처리 완료 상태로 변경."""
        self.status = DocumentStatus.PROCESSED
        self.processed_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_as_failed(self, error_message: str) -> None:
        """문서를 처리 실패 상태로 변경."""
        self.status = DocumentStatus.FAILED
        self.metadata["error"] = error_message
        self.updated_at = datetime.now()

    def add_connected_node(self, node_id: NodeId) -> None:
        """연결된 노드 추가."""
        if node_id not in self.connected_nodes:
            self.connected_nodes.append(node_id)
            self.updated_at = datetime.now()

    def add_connected_relationship(self, relationship_id: RelationshipId) -> None:
        """연결된 관계 추가."""
        if relationship_id not in self.connected_relationships:
            self.connected_relationships.append(relationship_id)
            self.updated_at = datetime.now()

    def remove_connected_node(self, node_id: NodeId) -> None:
        """연결된 노드 제거."""
        if node_id in self.connected_nodes:
            self.connected_nodes.remove(node_id)
            self.updated_at = datetime.now()

    def remove_connected_relationship(self, relationship_id: RelationshipId) -> None:
        """연결된 관계 제거."""
        if relationship_id in self.connected_relationships:
            self.connected_relationships.remove(relationship_id)
            self.updated_at = datetime.now()

    def get_word_count(self) -> int:
        """문서의 단어 수 계산."""
        return len(self.content.split())

    def get_char_count(self) -> int:
        """문서의 문자 수 계산."""
        return len(self.content)

    def is_processed(self) -> bool:
        """문서가 처리되었는지 확인."""
        return self.status == DocumentStatus.PROCESSED

    def has_connected_elements(self) -> bool:
        """연결된 노드나 관계가 있는지 확인."""
        return len(self.connected_nodes) > 0 or len(self.connected_relationships) > 0

    def update_metadata(self, key: str, value: Any) -> None:
        """메타데이터 업데이트."""
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def increment_version(self) -> None:
        """버전을 증가시킵니다."""
        self.version += 1
        self.updated_at = datetime.now()

    def get_version(self) -> int:
        """현재 버전을 반환합니다."""
        return self.version

    def set_version(self, version: int) -> None:
        """버전을 설정합니다."""
        self.version = version

    def __str__(self) -> str:
        return f"Document(id={self.id}, title='{self.title}', status={self.status.value})"

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id!r}, title={self.title!r}, "
            f"doc_type={self.doc_type.value}, status={self.status.value})"
        )
