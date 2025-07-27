"""
지식 검색 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.domain.services.knowledge_search import (
    SearchCriteria,
    SearchResultCollection,
    SearchStrategy,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId


class KnowledgeSearchUseCase(ABC):
    """지식 검색 Use Case 인터페이스."""

    @abstractmethod
    async def search_knowledge(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
    ) -> SearchResultCollection:
        """통합 지식 검색을 수행합니다."""
        pass

    @abstractmethod
    async def search_documents(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.5
    ) -> List[Document]:
        """문서 검색을 수행합니다."""
        pass

    @abstractmethod
    async def search_nodes(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[Node]:
        """노드 검색을 수행합니다."""
        pass

    @abstractmethod
    async def search_relationships(
        self,
        query: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[Relationship]:
        """관계 검색을 수행합니다."""
        pass

    @abstractmethod
    async def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> SearchResultCollection:
        """임베딩 기반 의미적 검색을 수행합니다."""
        pass


class KnowledgeNavigationUseCase(ABC):
    """지식 탐색 Use Case 인터페이스."""

    @abstractmethod
    async def find_related_documents(self, node_id: NodeId) -> List[Document]:
        """노드와 관련된 문서들을 찾습니다."""
        pass

    @abstractmethod
    async def find_connected_nodes(self, document_id: DocumentId) -> List[Node]:
        """문서와 연결된 노드들을 찾습니다."""
        pass

    @abstractmethod
    async def find_node_relationships(self, node_id: NodeId) -> List[Relationship]:
        """노드와 연결된 관계들을 찾습니다."""
        pass

    @abstractmethod
    async def get_knowledge_graph_for_document(self, document_id: DocumentId) -> dict:
        """문서와 관련된 지식 그래프를 조회합니다."""
        pass

    @abstractmethod
    async def get_search_suggestions(
        self, partial_query: str, limit: int = 10
    ) -> List[str]:
        """검색 자동완성 제안을 제공합니다."""
        pass
