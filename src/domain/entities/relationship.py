"""
관계 도메인 엔티티.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector


class RelationshipType(Enum):
    """관계 유형."""

    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    COLLABORATES_WITH = "collaborates_with"
    PART_OF = "part_of"
    LEADS = "leads"
    CREATES = "creates"
    USES = "uses"
    SIMILAR_TO = "similar_to"
    CAUSED_BY = "caused_by"
    INFLUENCES = "influences"
    CONTAINS = "contains"
    OTHER = "other"


@dataclass
class Relationship:
    """
    관계 도메인 엔티티.

    지식 그래프의 관계를 나타냅니다.
    두 노드 사이의 관계를 표현하고,
    원본 문서와의 연결 정보를 유지합니다.
    """

    id: RelationshipId
    source_node_id: NodeId
    target_node_id: NodeId
    relationship_type: RelationshipType
    label: str  # 관계를 설명하는 자연어 레이블
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 관계 추출 신뢰도 (0.0 ~ 1.0)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 문서 연결 정보
    source_documents: list[DocumentId] = field(default_factory=list)

    # 임베딩 정보
    embedding: Optional[Vector] = None
    embedding_model: Optional[str] = None
    embedding_created_at: Optional[datetime] = None

    # 추출 정보 (문서 내 컨텍스트, 문장 등)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label.strip():
            raise ValueError("Relationship label cannot be empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.source_node_id == self.target_node_id:
            raise ValueError("Source and target nodes cannot be the same")

    def add_source_document(
        self,
        document_id: DocumentId,
        context: Optional[str] = None,
        sentence: Optional[str] = None,
    ) -> None:
        """원본 문서를 추가합니다."""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
            if context or sentence:
                self.add_extraction_context(document_id, context, sentence)
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

    def add_extraction_context(
        self,
        document_id: DocumentId,
        context: Optional[str] = None,
        sentence: Optional[str] = None,
    ) -> None:
        """문서 내 추출 컨텍스트를 추가합니다."""
        self.extraction_metadata[f"context_{document_id}"] = {
            "context": context,
            "sentence": sentence,
            "extracted_at": datetime.now().isoformat(),
        }
        self.updated_at = datetime.now()

    def get_extraction_context(self, document_id: DocumentId) -> Optional[dict[str, str]]:
        """특정 문서의 추출 컨텍스트를 조회합니다."""
        return self.extraction_metadata.get(f"context_{document_id}")

    def set_embedding(self, embedding: Vector, model_name: str) -> None:
        """관계 임베딩을 설정합니다."""
        self.embedding = embedding
        self.embedding_model = model_name
        self.embedding_created_at = datetime.now()
        self.updated_at = datetime.now()

    def has_embedding(self) -> bool:
        """임베딩이 설정되었는지 확인합니다."""
        return self.embedding is not None

    def calculate_similarity(self, other: "Relationship") -> float:
        """다른 관계와의 유사도를 계산합니다."""
        if not self.has_embedding() or not other.has_embedding():
            raise ValueError("Both relationships must have embeddings for similarity calculation")

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

    def update_confidence(self, confidence: float) -> None:
        """신뢰도를 업데이트합니다."""
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = confidence
        self.updated_at = datetime.now()

    def get_all_contexts(self) -> dict[str, dict[str, str]]:
        """모든 문서의 추출 컨텍스트를 조회합니다."""
        contexts = {}
        for key, value in self.extraction_metadata.items():
            if key.startswith("context_"):
                doc_id = key.replace("context_", "")
                contexts[doc_id] = value
        return contexts

    def is_from_document(self, document_id: DocumentId) -> bool:
        """관계가 특정 문서에서 추출되었는지 확인합니다."""
        return document_id in self.source_documents

    def get_document_count(self) -> int:
        """연결된 문서의 수를 반환합니다."""
        return len(self.source_documents)

    def involves_node(self, node_id: NodeId) -> bool:
        """특정 노드가 이 관계에 포함되는지 확인합니다."""
        return node_id in (self.source_node_id, self.target_node_id)

    def get_other_node_id(self, node_id: NodeId) -> Optional[NodeId]:
        """주어진 노드에 대해 다른 노드 ID를 반환합니다."""
        if self.source_node_id == node_id:
            return self.target_node_id
        if self.target_node_id == node_id:
            return self.source_node_id
        return None

    def reverse(self) -> "Relationship":
        """방향이 반대인 새로운 관계를 생성합니다."""
        reversed_rel = Relationship(
            id=RelationshipId.generate(),
            source_node_id=self.target_node_id,
            target_node_id=self.source_node_id,
            relationship_type=self.relationship_type,
            label=f"reverse_of_{self.label}",
            properties=self.properties.copy(),
            confidence=self.confidence,
            source_documents=self.source_documents.copy(),
            embedding=self.embedding,
            embedding_model=self.embedding_model,
            embedding_created_at=self.embedding_created_at,
            extraction_metadata=self.extraction_metadata.copy(),
        )
        return reversed_rel

    def merge_with(self, other: "Relationship") -> None:
        """다른 관계와 병합합니다 (중복 제거)."""
        # 관계가 동일한 노드들 사이에 있는지 확인
        if not (
            (
                self.source_node_id == other.source_node_id
                and self.target_node_id == other.target_node_id
            )
            or (
                self.source_node_id == other.target_node_id
                and self.target_node_id == other.source_node_id
            )
        ):
            raise ValueError("Cannot merge relationships between different nodes")

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

        # 더 높은 신뢰도 사용
        self.confidence = max(self.confidence, other.confidence)

        # 더 나은 임베딩이 있으면 업데이트
        if not self.has_embedding() and other.has_embedding():
            self.embedding = other.embedding
            self.embedding_model = other.embedding_model
            self.embedding_created_at = other.embedding_created_at

        self.updated_at = datetime.now()

    def get_textual_representation(self) -> str:
        """관계의 텍스트 표현을 생성합니다 (임베딩용)."""
        return f"{self.label} (type: {self.relationship_type.value}, confidence: {self.confidence})"

    def __str__(self) -> str:
        return f"Relationship(id={self.id}, {self.source_node_id} --[{self.label}]--> {self.target_node_id})"

    def __repr__(self) -> str:
        return (
            f"Relationship(id={self.id!r}, source={self.source_node_id!r}, "
            f"target={self.target_node_id!r}, type={self.relationship_type.value}, "
            f"label={self.label!r}, confidence={self.confidence})"
        )
