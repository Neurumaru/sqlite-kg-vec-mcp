"""
노드 저장소 포트.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.entities.node import Node, NodeType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.vector import Vector


class NodeRepository(ABC):
    """
    노드 저장소 포트.

    노드의 영속성을 담당하는 인터페이스입니다.
    """

    @abstractmethod
    async def save(self, node: Node) -> Node:
        """
        노드를 저장합니다.

        Args:
            node: 저장할 노드

        Returns:
            저장된 노드
        """
        pass

    @abstractmethod
    async def find_by_id(self, node_id: NodeId) -> Optional[Node]:
        """
        ID로 노드를 찾습니다.

        Args:
            node_id: 노드 ID

        Returns:
            찾은 노드 또는 None
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> List[Node]:
        """
        이름으로 노드를 찾습니다.

        Args:
            name: 노드 이름

        Returns:
            매칭되는 노드들
        """
        pass

    @abstractmethod
    async def find_by_type(self, node_type: NodeType) -> List[Node]:
        """
        타입으로 노드를 찾습니다.

        Args:
            node_type: 노드 타입

        Returns:
            해당 타입의 노드들
        """
        pass

    @abstractmethod
    async def find_by_document(self, document_id: DocumentId) -> List[Node]:
        """
        특정 문서에서 추출된 노드들을 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            해당 문서에서 추출된 노드들
        """
        pass

    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Node]:
        """
        모든 노드를 조회합니다.

        Args:
            limit: 최대 반환 개수
            offset: 건너뛸 개수

        Returns:
            노드들
        """
        pass

    @abstractmethod
    async def search_by_properties(self, properties: Dict[str, Any]) -> List[Node]:
        """
        속성으로 노드를 검색합니다.

        Args:
            properties: 검색할 속성들

        Returns:
            매칭되는 노드들
        """
        pass

    @abstractmethod
    async def find_similar_nodes(
        self, query_vector: Vector, similarity_threshold: float = 0.5, limit: int = 10
    ) -> List[tuple[Node, float]]:
        """
        벡터 유사도로 노드를 찾습니다.

        Args:
            query_vector: 쿼리 벡터
            similarity_threshold: 유사도 임계값
            limit: 최대 반환 개수

        Returns:
            (노드, 유사도 점수) 튜플들
        """
        pass

    @abstractmethod
    async def find_nodes_with_embedding(self) -> List[Node]:
        """
        임베딩이 있는 노드들을 찾습니다.

        Returns:
            임베딩이 있는 노드들
        """
        pass

    @abstractmethod
    async def find_nodes_without_embedding(self) -> List[Node]:
        """
        임베딩이 없는 노드들을 찾습니다.

        Returns:
            임베딩이 없는 노드들
        """
        pass

    @abstractmethod
    async def update(self, node: Node) -> Node:
        """
        노드를 업데이트합니다.

        Args:
            node: 업데이트할 노드

        Returns:
            업데이트된 노드
        """
        pass

    @abstractmethod
    async def delete(self, node_id: NodeId) -> bool:
        """
        노드를 삭제합니다.

        Args:
            node_id: 삭제할 노드 ID

        Returns:
            삭제 성공 여부
        """
        pass

    @abstractmethod
    async def exists(self, node_id: NodeId) -> bool:
        """
        노드가 존재하는지 확인합니다.

        Args:
            node_id: 확인할 노드 ID

        Returns:
            존재 여부
        """
        pass

    @abstractmethod
    async def count_by_type(self, node_type: NodeType) -> int:
        """
        타입별 노드 개수를 반환합니다.

        Args:
            node_type: 노드 타입

        Returns:
            해당 타입의 노드 개수
        """
        pass

    @abstractmethod
    async def count_total(self) -> int:
        """
        전체 노드 개수를 반환합니다.

        Returns:
            전체 노드 개수
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Node]:
        """
        날짜 범위로 노드를 찾습니다.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            해당 기간에 생성된 노드들
        """
        pass

    @abstractmethod
    async def find_nodes_by_name_pattern(self, pattern: str) -> List[Node]:
        """
        이름 패턴으로 노드를 찾습니다.

        Args:
            pattern: 검색 패턴

        Returns:
            패턴에 매칭되는 노드들
        """
        pass

    @abstractmethod
    async def batch_save(self, nodes: List[Node]) -> List[Node]:
        """
        여러 노드를 일괄 저장합니다.

        Args:
            nodes: 저장할 노드들

        Returns:
            저장된 노드들
        """
        pass

    @abstractmethod
    async def batch_update_embeddings(
        self, node_embeddings: Dict[NodeId, Vector], model_name: str
    ) -> int:
        """
        여러 노드의 임베딩을 일괄 업데이트합니다.

        Args:
            node_embeddings: 노드 ID -> 임베딩 매핑
            model_name: 임베딩 모델명

        Returns:
            업데이트된 노드 개수
        """
        pass

    @abstractmethod
    async def find_orphaned_nodes(self) -> List[Node]:
        """
        연결된 관계가 없는 고립된 노드들을 찾습니다.

        Returns:
            고립된 노드들
        """
        pass
