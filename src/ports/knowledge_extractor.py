"""
지식 추출 포트.
"""

from abc import ABC, abstractmethod

from src.dto import DocumentData, NodeData, RelationshipData


class KnowledgeExtractor(ABC):
    """
    지식 추출 포트.

    문서로부터 노드(개체)와 관계를 추출하는 기능을 제공합니다.
    """

    @abstractmethod
    async def extract(
        self, document: DocumentData
    ) -> tuple[list[NodeData], list[RelationshipData]]:
        """
        문서에서 지식(노드와 관계)을 추출합니다.

        Args:
            document: 분석할 문서 데이터

        Returns:
            (노드 데이터 리스트, 관계 데이터 리스트) 튜플
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """
        지식 추출 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
