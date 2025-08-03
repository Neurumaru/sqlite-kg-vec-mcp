"""
지식 검색 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod

from src.config.search_config import DEFAULT_SIMILARITY_THRESHOLD
from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.domain.services.knowledge_search import (
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
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
    ) -> SearchResultCollection:
        """통합 지식 검색을 수행합니다.

        인자:
            query: 검색 쿼리 (비어있지 않아야 함)
            strategy: 검색 전략
            limit: 최대 결과 수 (1 이상)
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
            include_documents: 문서 결과 포함 여부
            include_nodes: 노드 결과 포함 여부
            include_relationships: 관계 결과 포함 여부

        반환:
            검색 결과 컬렉션

        예외:
            ValueError: 매개변수가 유효하지 않은 경우
        """

    @abstractmethod
    async def search_documents(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.5
    ) -> list[Document]:
        """문서 검색을 수행합니다."""

    @abstractmethod
    async def search_nodes(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[Node]:
        """노드 검색을 수행합니다."""

    @abstractmethod
    async def search_relationships(
        self,
        query: str,
        relationship_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[Relationship]:
        """관계 검색을 수행합니다."""

    @abstractmethod
    async def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> SearchResultCollection:
        """임베딩 기반 의미적 검색을 수행합니다."""


class KnowledgeNavigationUseCase(ABC):
    """지식 탐색 Use Case 인터페이스."""

    @abstractmethod
    async def find_related_documents(self, node_id: NodeId) -> list[Document]:
        """노드와 관련된 문서들을 찾습니다."""

    @abstractmethod
    async def find_connected_nodes(self, document_id: DocumentId) -> list[Node]:
        """문서와 연결된 노드들을 찾습니다."""

    @abstractmethod
    async def find_node_relationships(self, node_id: NodeId) -> list[Relationship]:
        """노드와 연결된 관계들을 찾습니다."""

    @abstractmethod
    async def get_knowledge_graph_for_document(self, document_id: DocumentId) -> dict:
        """문서와 관련된 지식 그래프를 조회합니다."""

    @abstractmethod
    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> list[str]:
        """검색 자동완성 제안을 제공합니다."""
