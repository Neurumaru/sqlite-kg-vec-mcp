"""
노드 도메인 엔티티.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.vector import Vector


class NodeType(Enum):
    """노드 유형."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    OTHER = "other"


@dataclass
class Node:
    """
    노드 도메인 엔티티.

    지식 그래프의 노드를 나타냅니다.
    문서에서 추출된 엔티티를 표현하고,
    원본 문서와의 연결 정보를 유지합니다.
    """

    id: NodeId
    name: str
    node_type: NodeType
    description: Optional[str] = None
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 문서 연결 정보
    source_documents: list[DocumentId] = field(default_factory=list)

    # 임베딩 정보
    embedding: Optional[Vector] = None
    embedding_model: Optional[str] = None
    embedding_created_at: Optional[datetime] = None

    # 추출 정보 (문서 내 위치, 컨텍스트 등)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Node name cannot be empty")

    def add_source_document(self, document_id: Optional[DocumentId, context: Optional[str] = None) -> None:
        """원본 문서를 추가합니다."""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
            if context:
                self.add_extraction_context(document_id, context)
            self.updated_at = datetime.now()

    def remove_source_document(self, document_id: DocumentId) -> None:
        """원본 문서를 제거합니다."""
        if document_id in self.source_documents:
            self.source_documents.remove(document_id)
            # 관련 추출 메타데이터도 제거
            key_to_remove = f"context_{document_id}"
            if key_to_remove in self.extraction_metadata:
                del self.extraction_metadata[key_to_remove]
            self.updated_at = datetime.now()

    def add_extraction_context(self, document_id: DocumentId, context: str) -> None:
        """문서 내 추출 컨텍스트를 추가합니다."""
        self.extraction_metadata[f"context_{document_id}"] = {
            "context": context,
            "extracted_at": datetime.now().isoformat(),
        }
        self.updated_at = datetime.now()

    def get_extraction_context(self, document_id: DocumentId) -> Optional[str]:
        """특정 문서의 추출 컨텍스트를 조회합니다."""
        context_data = self.extraction_metadata.get(f"context_{document_id}")
        return context_data.get("context") if context_data else None

    def set_embedding(self, embedding: Vector, model_name: str) -> None:
        """노드 임베딩을 설정합니다."""
        self.embedding = embedding
        self.embedding_model = model_name
        self.embedding_created_at = datetime.now()
        self.updated_at = datetime.now()

    def has_embedding(self) -> bool:
        """임베딩이 설정되었는지 확인합니다."""
        return self.embedding is not None

    def calculate_similarity(self, other: "Node") -> float:
        """다른 노드와의 유사도를 계산합니다."""
        if not self.has_embedding() or not other.has_embedding():
            raise ValueError("Both nodes must have embeddings for similarity calculation")

        # mypy: 임베딩이 None이 아님을 확인했으므로 assert 사용
        assert self.embedding is not None and other.embedding is not None
        return self.embedding.cosine_similarity(other.embedding)

    def update_property(self, key: str, value: Any) -> None:
        """속성을 업데이트합니다."""
        self.properties[key] = value
        self.updated_at = datetime.now()

    def remove_property(self, key: str) -> None:
        """속성을 제거합니다."""
        if key in self.properties:
            del self.properties[key]
            self.updated_at = datetime.now()

    def get_all_contexts(self) -> dict[str, str]:
        """모든 문서의 추출 컨텍스트를 조회합니다."""
        contexts = {}
        for key, value in self.extraction_metadata.items():
            if key.startswith("context_"):
                doc_id = key.replace("context_", "")
                contexts[doc_id] = value.get("context", "")
        return contexts

    def is_from_document(self, document_id: DocumentId) -> bool:
        """노드가 특정 문서에서 추출되었는지 확인합니다."""
        return document_id in self.source_documents

    def get_document_count(self) -> int:
        """연결된 문서의 수를 반환합니다."""
        return len(self.source_documents)

    def merge_with(self, other: "Node") -> None:
        """다른 노드와 병합합니다 (중복 제거)."""
        # 원본 문서 병합
        for doc_id in other.source_documents:
            if doc_id not in self.source_documents:
                self.source_documents.append(doc_id)

        # 속성 병합 (기존 속성 우선)
        for key, value in other.properties.items():
            if key not in self.properties:
                self.properties[key] = value

        # 추출 메타데이터 병합
        for key, value in other.extraction_metadata.items():
            if key not in self.extraction_metadata:
                self.extraction_metadata[key] = value

        # 더 나은 임베딩이 있으면 업데이트
        if not self.has_embedding() and other.has_embedding():
            self.embedding = other.embedding
            self.embedding_model = other.embedding_model
            self.embedding_created_at = other.embedding_created_at

        self.updated_at = datetime.now()

    def __str__(self) -> str:
        return f"Node(id={self.id}, name='{self.name}', type={self.node_type.value})"

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id!r}, name={self.name!r}, "
            f"node_type={self.node_type.value}, "
            f"source_docs={len(self.source_documents)})"
        )
