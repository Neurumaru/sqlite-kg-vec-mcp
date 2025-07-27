"""
관계 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector


class RelationshipManagementUseCase(ABC):
    """관계 관리 Use Case 인터페이스."""

    @abstractmethod
    async def create_relationship(
        self,
        source_node_id: NodeId,
        target_node_id: NodeId,
        relationship_type: RelationshipType,
        label: str,
        properties: Optional[dict] = None,
        weight: float = 1.0
    ) -> Relationship:
        """새 관계를 생성합니다."""
        pass

    @abstractmethod
    async def get_relationship(self, relationship_id: RelationshipId) -> Optional[Relationship]:
        """관계를 조회합니다."""
        pass

    @abstractmethod
    async def list_relationships(
        self,
        relationship_type: Optional[RelationshipType] = None,
        source_node_id: Optional[NodeId] = None,
        target_node_id: Optional[NodeId] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Relationship]:
        """관계 목록을 조회합니다."""
        pass

    @abstractmethod
    async def update_relationship(
        self,
        relationship_id: RelationshipId,
        label: Optional[str] = None,
        properties: Optional[dict] = None,
        weight: Optional[float] = None
    ) -> Relationship:
        """관계를 업데이트합니다."""
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: RelationshipId) -> bool:
        """관계를 삭제합니다."""
        pass

    @abstractmethod
    async def get_node_relationships(
        self, node_id: NodeId, direction: str = "both"
    ) -> List[Relationship]:
        """노드의 관계들을 조회합니다 (방향: in, out, both)."""
        pass


class RelationshipAnalysisUseCase(ABC):
    """관계 분석 Use Case 인터페이스."""

    @abstractmethod
    async def find_shortest_path(
        self, source_node_id: NodeId, target_node_id: NodeId, max_depth: int = 5
    ) -> Optional[List[Relationship]]:
        """두 노드 간의 최단 경로를 찾습니다."""
        pass

    @abstractmethod
    async def get_node_neighbors(
        self, node_id: NodeId, depth: int = 1
    ) -> List[NodeId]:
        """노드의 이웃 노드들을 조회합니다."""
        pass

    @abstractmethod
    async def calculate_relationship_strength(
        self, source_node_id: NodeId, target_node_id: NodeId
    ) -> float:
        """두 노드 간의 관계 강도를 계산합니다."""
        pass

    @abstractmethod
    async def find_influential_nodes(
        self, limit: int = 10
    ) -> List[tuple[NodeId, float]]:
        """영향력 있는 노드들을 찾습니다."""
        pass

    @abstractmethod
    async def cluster_nodes(
        self, algorithm: str = "community"
    ) -> dict[str, List[NodeId]]:
        """노드들을 클러스터링합니다."""
        pass


class RelationshipEmbeddingUseCase(ABC):
    """관계 임베딩 관련 Use Case 인터페이스."""

    @abstractmethod
    async def generate_relationship_embedding(self, relationship_id: RelationshipId) -> Vector:
        """관계의 임베딩을 생성합니다."""
        pass

    @abstractmethod
    async def update_relationship_embedding(
        self, relationship_id: RelationshipId, embedding: Vector
    ) -> bool:
        """관계의 임베딩을 업데이트합니다."""
        pass

    @abstractmethod
    async def get_relationship_embedding(
        self, relationship_id: RelationshipId
    ) -> Optional[Vector]:
        """관계의 임베딩을 조회합니다."""
        pass

    @abstractmethod
    async def find_similar_relationships(
        self, relationship_id: RelationshipId, limit: int = 10, threshold: float = 0.7
    ) -> List[tuple[Relationship, float]]:
        """유사한 관계들을 찾습니다."""
        pass