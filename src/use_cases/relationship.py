"""
관계 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod

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
        properties: dict | None = None,
        weight: float = 1.0,
    ) -> Relationship:
        """새 관계를 생성합니다."""

    @abstractmethod
    async def get_relationship(self, relationship_id: RelationshipId) -> Relationship | None:
        """관계를 조회합니다."""

    @abstractmethod
    async def list_relationships(
        self,
        relationship_type: RelationshipType | None = None,
        source_node_id: NodeId | None = None,
        target_node_id: NodeId | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Relationship]:
        """관계 목록을 조회합니다."""

    @abstractmethod
    async def update_relationship(
        self,
        relationship_id: RelationshipId,
        label: str | None = None,
        properties: dict | None = None,
        weight: float | None = None,
    ) -> Relationship:
        """관계를 업데이트합니다."""

    @abstractmethod
    async def delete_relationship(self, relationship_id: RelationshipId) -> bool:
        """관계를 삭제합니다."""

    @abstractmethod
    async def get_node_relationships(
        self, node_id: NodeId, direction: str = "both"
    ) -> list[Relationship]:
        """노드의 관계들을 조회합니다 (방향: in, out, both)."""


class RelationshipAnalysisUseCase(ABC):
    """관계 분석 Use Case 인터페이스."""

    @abstractmethod
    async def find_shortest_path(
        self, source_node_id: NodeId, target_node_id: NodeId, max_depth: int = 5
    ) -> list[Relationship] | None:
        """두 노드 간의 최단 경로를 찾습니다."""

    @abstractmethod
    async def get_node_neighbors(self, node_id: NodeId, depth: int = 1) -> list[NodeId]:
        """노드의 이웃 노드들을 조회합니다."""

    @abstractmethod
    async def calculate_relationship_strength(
        self, source_node_id: NodeId, target_node_id: NodeId
    ) -> float:
        """두 노드 간의 관계 강도를 계산합니다."""

    @abstractmethod
    async def find_influential_nodes(self, limit: int = 10) -> list[tuple[NodeId, float]]:
        """영향력 있는 노드들을 찾습니다."""

    @abstractmethod
    async def cluster_nodes(self, algorithm: str = "community") -> dict[str, list[NodeId]]:
        """노드들을 클러스터링합니다."""


class RelationshipEmbeddingUseCase(ABC):
    """관계 임베딩 관련 Use Case 인터페이스."""

    @abstractmethod
    async def generate_relationship_embedding(self, relationship_id: RelationshipId) -> Vector:
        """관계의 임베딩을 생성합니다."""

    @abstractmethod
    async def update_relationship_embedding(
        self, relationship_id: RelationshipId, embedding: Vector
    ) -> bool:
        """관계의 임베딩을 업데이트합니다."""

    @abstractmethod
    async def get_relationship_embedding(self, relationship_id: RelationshipId) -> Vector | None:
        """관계의 임베딩을 조회합니다."""

    @abstractmethod
    async def find_similar_relationships(
        self, relationship_id: RelationshipId, limit: int = 10, threshold: float = 0.7
    ) -> list[tuple[Relationship, float]]:
        """유사한 관계들을 찾습니다."""

    @abstractmethod
    async def batch_create_relationships(
        self,
        relationship_data: list[tuple[NodeId, NodeId, RelationshipType, str, dict | None, float]],
    ) -> list[Relationship]:
        """여러 관계를 일괄 생성합니다.

        Args:
            relationship_data: (source_id, target_id, type, label, properties, weight) 튜플들의 목록

        Returns:
            생성된 관계 목록

        Raises:
            ValueError: 입력 데이터가 잘못된 경우
        """

    @abstractmethod
    async def batch_analyze_paths(
        self, node_pairs: list[tuple[NodeId, NodeId]], max_depth: int = 5
    ) -> dict[tuple[NodeId, NodeId], list[Relationship] | None]:
        """여러 노드 쌍 간의 경로를 일괄 분석합니다.

        Args:
            node_pairs: 분석할 노드 쌍 목록
            max_depth: 최대 경로 깊이

        Returns:
            노드 쌍별 최단 경로 딕셔너리

        Raises:
            ValueError: 입력 데이터가 잘못된 경우
        """
