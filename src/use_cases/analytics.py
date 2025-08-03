"""
분석 및 통계 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId


class KnowledgeGraphAnalyticsUseCase(ABC):
    """지식 그래프 분석 Use Case 인터페이스."""

    @abstractmethod
    async def get_graph_statistics(self) -> dict[str, Any]:
        """지식 그래프 전체 통계를 조회합니다."""

    @abstractmethod
    async def get_document_statistics(self) -> dict[str, Any]:
        """문서 처리 통계를 조회합니다."""

    @abstractmethod
    async def get_node_statistics(self) -> dict[str, Any]:
        """노드 통계를 조회합니다."""

    @abstractmethod
    async def get_relationship_statistics(self) -> dict[str, Any]:
        """관계 통계를 조회합니다."""

    @abstractmethod
    async def get_embedding_statistics(self) -> dict[str, Any]:
        """임베딩 통계를 조회합니다."""

    @abstractmethod
    async def analyze_document_processing_performance(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """문서 처리 성능을 분석합니다."""

    @abstractmethod
    async def get_most_connected_nodes(self, limit: int = 10) -> list[tuple[NodeId, int]]:
        """가장 많이 연결된 노드들을 조회합니다.

        Args:
            limit: 조회할 노드 수 (1 이상이어야 함)

        Returns:
            (노드ID, 연결 수) 튜플의 리스트

        Raises:
            ValueError: limit이 1보다 작은 경우
        """

    @abstractmethod
    async def get_most_referenced_documents(self, limit: int = 10) -> list[tuple[DocumentId, int]]:
        """가장 많이 참조된 문서들을 조회합니다.

        Args:
            limit: 조회할 문서 수 (1 이상이어야 함)

        Returns:
            (문서ID, 참조 수) 튜플의 리스트

        Raises:
            ValueError: limit이 1보다 작은 경우
        """


class SearchAnalyticsUseCase(ABC):
    """검색 분석 Use Case 인터페이스."""

    @abstractmethod
    async def record_search_query(
        self, query: str, strategy: str, result_count: int, execution_time_ms: float
    ) -> None:
        """검색 쿼리를 기록합니다."""

    @abstractmethod
    async def get_popular_search_terms(
        self, limit: int = 10, period_days: int = 30
    ) -> list[tuple[str, int]]:
        """인기 검색어를 조회합니다."""

    @abstractmethod
    async def get_search_performance_metrics(self, period_days: int = 7) -> dict[str, Any]:
        """검색 성능 지표를 조회합니다."""

    @abstractmethod
    async def analyze_search_patterns(self, period_days: int = 30) -> dict[str, Any]:
        """검색 패턴을 분석합니다."""


class QualityAnalyticsUseCase(ABC):
    """품질 분석 Use Case 인터페이스."""

    @abstractmethod
    async def analyze_knowledge_completeness(self) -> dict[str, Any]:
        """지식의 완성도를 분석합니다."""

    @abstractmethod
    async def detect_duplicate_nodes(
        self, similarity_threshold: float = 0.9
    ) -> list[tuple[NodeId, NodeId, float]]:
        """중복 노드를 탐지합니다.

        Args:
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0 사이)

        Returns:
            (노드ID1, 노드ID2, 유사도) 튜플의 리스트

        Raises:
            ValueError: threshold가 0.0~1.0 범위를 벗어난 경우
        """

    @abstractmethod
    async def detect_orphaned_nodes(self) -> list[NodeId]:
        """고립된 노드들을 탐지합니다."""

    @abstractmethod
    async def analyze_embedding_quality(self) -> dict[str, Any]:
        """임베딩 품질을 분석합니다."""

    @abstractmethod
    async def batch_analyze_documents(
        self, document_ids: list[DocumentId], analysis_types: list[str]
    ) -> dict[DocumentId, dict[str, Any]]:
        """여러 문서에 대한 일괄 분석을 수행합니다.

        Args:
            document_ids: 분석할 문서 ID 목록
            analysis_types: 수행할 분석 유형 목록

        Returns:
            문서 ID별 분석 결과 딕셔너리

        Raises:
            ValueError: 입력 데이터가 잘못된 경우
        """

    @abstractmethod
    async def validate_relationship_consistency(self) -> dict[str, Any]:
        """관계의 일관성을 검증합니다."""
