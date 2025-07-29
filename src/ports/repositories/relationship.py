"""
간소화된 관계 저장소 포트.

실제 사용 현황 분석 결과 현재 사용되지 않는 인터페이스이므로
필요한 경우에만 최소한의 기능을 제공하도록 간소화되었습니다.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.dto import RelationshipData


class RelationshipRepository(ABC):
    """
    간소화된 관계 저장소 포트.

    현재 사용되지 않아 핵심 CRUD 기능만 제공합니다.
    필요시 추가 메서드를 점진적으로 확장할 수 있습니다.
    """

    @abstractmethod
    async def save(self, relationship: RelationshipData) -> RelationshipData:
        """
        관계를 저장합니다.

        Args:
            relationship: 저장할 관계

        Returns:
            저장된 관계
        """

    @abstractmethod
    async def find_by_id(self, relationship_id: str) -> Optional[RelationshipData]:
        """
        ID로 관계를 찾습니다.

        Args:
            relationship_id: 관계 ID

        Returns:
            찾은 관계 또는 None
        """

    @abstractmethod
    async def find_by_nodes(
        self, source_node_id: str, target_node_id: str
    ) -> List[RelationshipData]:
        """
        두 노드 간의 관계들을 찾습니다.

        Args:
            source_node_id: 소스 노드 ID
            target_node_id: 타겟 노드 ID

        Returns:
            두 노드 간의 관계들
        """

    @abstractmethod
    async def find_by_document(self, document_id: str) -> List[RelationshipData]:
        """
        특정 문서에서 추출된 관계들을 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            해당 문서에서 추출된 관계들
        """

    @abstractmethod
    async def update(self, relationship: RelationshipData) -> RelationshipData:
        """
        관계를 업데이트합니다.

        Args:
            relationship: 업데이트할 관계

        Returns:
            업데이트된 관계
        """

    @abstractmethod
    async def delete(self, relationship_id: str) -> bool:
        """
        관계를 삭제합니다.

        Args:
            relationship_id: 삭제할 관계 ID

        Returns:
            삭제 성공 여부
        """
