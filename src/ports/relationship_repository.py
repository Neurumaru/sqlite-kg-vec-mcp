"""
관계 저장소 포트.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector


class RelationshipRepository(ABC):
    """
    관계 저장소 포트.

    관계의 영속성을 담당하는 인터페이스입니다.
    """

    @abstractmethod
    async def save(self, relationship: Relationship) -> Relationship:
        """
        관계를 저장합니다.

        Args:
            relationship: 저장할 관계

        Returns:
            저장된 관계
        """
        pass

    @abstractmethod
    async def find_by_id(
        self, relationship_id: RelationshipId
    ) -> Optional[Relationship]:
        """
        ID로 관계를 찾습니다.

        Args:
            relationship_id: 관계 ID

        Returns:
            찾은 관계 또는 None
        """
        pass

    @abstractmethod
    async def find_by_nodes(
        self, source_node_id: NodeId, target_node_id: NodeId
    ) -> List[Relationship]:
        """
        두 노드 간의 관계들을 찾습니다.

        Args:
            source_node_id: 소스 노드 ID
            target_node_id: 타겟 노드 ID

        Returns:
            두 노드 간의 관계들
        """
        pass

    @abstractmethod
    async def find_by_source_node(self, node_id: NodeId) -> List[Relationship]:
        """
        특정 노드에서 출발하는 관계들을 찾습니다.

        Args:
            node_id: 소스 노드 ID

        Returns:
            해당 노드에서 출발하는 관계들
        """
        pass

    @abstractmethod
    async def find_by_target_node(self, node_id: NodeId) -> List[Relationship]:
        """
        특정 노드로 들어오는 관계들을 찾습니다.

        Args:
            node_id: 타겟 노드 ID

        Returns:
            해당 노드로 들어오는 관계들
        """
        pass

    @abstractmethod
    async def find_by_node(self, node_id: NodeId) -> List[Relationship]:
        """
        특정 노드와 연결된 모든 관계들을 찾습니다.

        Args:
            node_id: 노드 ID

        Returns:
            해당 노드와 연결된 모든 관계들
        """
        pass

    @abstractmethod
    async def find_by_type(
        self, relationship_type: RelationshipType
    ) -> List[Relationship]:
        """
        타입으로 관계를 찾습니다.

        Args:
            relationship_type: 관계 타입

        Returns:
            해당 타입의 관계들
        """
        pass

    @abstractmethod
    async def find_by_document(self, document_id: DocumentId) -> List[Relationship]:
        """
        특정 문서에서 추출된 관계들을 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            해당 문서에서 추출된 관계들
        """
        pass

    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Relationship]:
        """
        모든 관계를 조회합니다.

        Args:
            limit: 최대 반환 개수
            offset: 건너뛸 개수

        Returns:
            관계들
        """
        pass

    @abstractmethod
    async def find_by_confidence_range(
        self, min_confidence: float, max_confidence: float
    ) -> List[Relationship]:
        """
        신뢰도 범위로 관계를 찾습니다.

        Args:
            min_confidence: 최소 신뢰도
            max_confidence: 최대 신뢰도

        Returns:
            해당 신뢰도 범위의 관계들
        """
        pass

    @abstractmethod
    async def find_similar_relationships(
        self, query_vector: Vector, similarity_threshold: float = 0.5, limit: int = 10
    ) -> List[Tuple[Relationship, float]]:
        """
        벡터 유사도로 관계를 찾습니다.

        Args:
            query_vector: 쿼리 벡터
            similarity_threshold: 유사도 임계값
            limit: 최대 반환 개수

        Returns:
            (관계, 유사도 점수) 튜플들
        """
        pass

    @abstractmethod
    async def find_relationships_with_embedding(self) -> List[Relationship]:
        """
        임베딩이 있는 관계들을 찾습니다.

        Returns:
            임베딩이 있는 관계들
        """
        pass

    @abstractmethod
    async def find_relationships_without_embedding(self) -> List[Relationship]:
        """
        임베딩이 없는 관계들을 찾습니다.

        Returns:
            임베딩이 없는 관계들
        """
        pass

    @abstractmethod
    async def update(self, relationship: Relationship) -> Relationship:
        """
        관계를 업데이트합니다.

        Args:
            relationship: 업데이트할 관계

        Returns:
            업데이트된 관계
        """
        pass

    @abstractmethod
    async def delete(self, relationship_id: RelationshipId) -> bool:
        """
        관계를 삭제합니다.

        Args:
            relationship_id: 삭제할 관계 ID

        Returns:
            삭제 성공 여부
        """
        pass

    @abstractmethod
    async def exists(self, relationship_id: RelationshipId) -> bool:
        """
        관계가 존재하는지 확인합니다.

        Args:
            relationship_id: 확인할 관계 ID

        Returns:
            존재 여부
        """
        pass

    @abstractmethod
    async def count_by_type(self, relationship_type: RelationshipType) -> int:
        """
        타입별 관계 개수를 반환합니다.

        Args:
            relationship_type: 관계 타입

        Returns:
            해당 타입의 관계 개수
        """
        pass

    @abstractmethod
    async def count_total(self) -> int:
        """
        전체 관계 개수를 반환합니다.

        Returns:
            전체 관계 개수
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Relationship]:
        """
        날짜 범위로 관계를 찾습니다.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            해당 기간에 생성된 관계들
        """
        pass

    @abstractmethod
    async def find_by_label_pattern(self, pattern: str) -> List[Relationship]:
        """
        레이블 패턴으로 관계를 찾습니다.

        Args:
            pattern: 검색 패턴

        Returns:
            패턴에 매칭되는 관계들
        """
        pass

    @abstractmethod
    async def batch_save(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        여러 관계를 일괄 저장합니다.

        Args:
            relationships: 저장할 관계들

        Returns:
            저장된 관계들
        """
        pass

    @abstractmethod
    async def batch_update_embeddings(
        self, relationship_embeddings: Dict[RelationshipId, Vector], model_name: str
    ) -> int:
        """
        여러 관계의 임베딩을 일괄 업데이트합니다.

        Args:
            relationship_embeddings: 관계 ID -> 임베딩 매핑
            model_name: 임베딩 모델명

        Returns:
            업데이트된 관계 개수
        """
        pass

    @abstractmethod
    async def find_paths_between_nodes(
        self, source_id: NodeId, target_id: NodeId, max_depth: int = 3
    ) -> List[List[Relationship]]:
        """
        두 노드 간의 경로를 찾습니다.

        Args:
            source_id: 시작 노드 ID
            target_id: 목표 노드 ID
            max_depth: 최대 탐색 깊이

        Returns:
            경로들 (각 경로는 관계들의 리스트)
        """
        pass

    @abstractmethod
    async def get_node_degree(self, node_id: NodeId) -> Dict[str, int]:
        """
        노드의 차수 정보를 반환합니다.

        Args:
            node_id: 노드 ID

        Returns:
            {"in_degree": int, "out_degree": int, "total_degree": int}
        """
        pass
