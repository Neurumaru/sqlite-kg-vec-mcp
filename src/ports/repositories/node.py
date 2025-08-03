"""
간소화된 노드 저장소 포트.

실제 사용 현황 분석 결과 현재 사용되지 않는 인터페이스이므로
필요한 경우에만 최소한의 기능을 제공하도록 간소화되었습니다.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.dto import NodeData


class NodeRepository(ABC):
    """
    간소화된 노드 저장소 포트.

    현재 사용되지 않아 핵심 CRUD 기능만 제공합니다.
    필요시 추가 메서드를 점진적으로 확장할 수 있습니다.
    """

    @abstractmethod
    async def save(self, node: NodeData) -> NodeData:
        """
        노드를 저장합니다.

        Args:
            node: 저장할 노드

        Returns:
            저장된 노드
        """

    @abstractmethod
    async def find_by_id(self, node_id: str) -> Optional[NodeData]:
        """
        ID로 노드를 찾습니다.

        Args:
            node_id: 노드 ID

        Returns:
            찾은 노드 또는 None
        """

    @abstractmethod
    async def find_by_document(self, document_id: str) -> list[NodeData]:
        """
        특정 문서에서 추출된 노드들을 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            해당 문서에서 추출된 노드들
        """

    @abstractmethod
    async def update(self, node: NodeData) -> NodeData:
        """
        노드를 업데이트합니다.

        Args:
            node: 업데이트할 노드

        Returns:
            업데이트된 노드
        """

    @abstractmethod
    async def delete(self, node_id: str) -> bool:
        """
        노드를 삭제합니다.

        Args:
            node_id: 삭제할 노드 ID

        Returns:
            삭제 성공 여부
        """
