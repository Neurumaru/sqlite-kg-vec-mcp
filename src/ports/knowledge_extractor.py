"""
지식 추출 포트.
"""

from abc import ABC, abstractmethod
from typing import List

from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship


class KnowledgeExtractor(ABC):
    """
    지식 추출 포트.
    
    문서로부터 노드(개체)와 관계를 추출하는 기능을 제공합니다.
    """

    @abstractmethod
    async def extract_entities(self, document: Document) -> List[Node]:
        """
        문서에서 개체(노드)를 추출합니다.

        Args:
            document: 분석할 문서

        Returns:
            추출된 노드 리스트
        """
        pass

    @abstractmethod
    async def extract_relationships(
        self, document: Document, nodes: List[Node]
    ) -> List[Relationship]:
        """
        문서에서 관계를 추출합니다.

        Args:
            document: 분석할 문서
            nodes: 문서에서 추출된 노드들

        Returns:
            추출된 관계 리스트
        """
        pass

    @abstractmethod
    async def extract_knowledge(
        self, document: Document
    ) -> tuple[List[Node], List[Relationship]]:
        """
        문서에서 전체 지식(노드와 관계)을 추출합니다.

        Args:
            document: 분석할 문서

        Returns:
            (노드 리스트, 관계 리스트) 튜플
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        지식 추출 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
        pass