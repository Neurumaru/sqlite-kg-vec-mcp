"""
분석 및 통계 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId


class KnowledgeGraphAnalyticsUseCase(ABC):
    """지식 그래프 분석 Use Case 인터페이스."""

    @abstractmethod
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """지식 그래프 전체 통계를 조회합니다."""
        pass

    @abstractmethod
    async def get_document_statistics(self) -> Dict[str, Any]:
        """문서 처리 통계를 조회합니다."""
        pass

    @abstractmethod
    async def get_node_statistics(self) -> Dict[str, Any]:
        """노드 통계를 조회합니다."""
        pass

    @abstractmethod
    async def get_relationship_statistics(self) -> Dict[str, Any]:
        """관계 통계를 조회합니다."""
        pass

    @abstractmethod
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """임베딩 통계를 조회합니다."""
        pass

    @abstractmethod
    async def analyze_document_processing_performance(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """문서 처리 성능을 분석합니다."""
        pass

    @abstractmethod
    async def get_most_connected_nodes(
        self, limit: int = 10
    ) -> List[tuple[NodeId, int]]:
        """가장 많이 연결된 노드들을 조회합니다."""
        pass

    @abstractmethod
    async def get_most_referenced_documents(
        self, limit: int = 10
    ) -> List[tuple[DocumentId, int]]:
        """가장 많이 참조된 문서들을 조회합니다."""
        pass


class SearchAnalyticsUseCase(ABC):
    """검색 분석 Use Case 인터페이스."""

    @abstractmethod
    async def record_search_query(
        self, query: str, strategy: str, result_count: int, execution_time_ms: float
    ) -> None:
        """검색 쿼리를 기록합니다."""
        pass

    @abstractmethod
    async def get_popular_search_terms(
        self, limit: int = 10, period_days: int = 30
    ) -> List[tuple[str, int]]:
        """인기 검색어를 조회합니다."""
        pass

    @abstractmethod
    async def get_search_performance_metrics(
        self, period_days: int = 7
    ) -> Dict[str, Any]:
        """검색 성능 지표를 조회합니다."""
        pass

    @abstractmethod
    async def analyze_search_patterns(self, period_days: int = 30) -> Dict[str, Any]:
        """검색 패턴을 분석합니다."""
        pass


class QualityAnalyticsUseCase(ABC):
    """품질 분석 Use Case 인터페이스."""

    @abstractmethod
    async def analyze_knowledge_completeness(self) -> Dict[str, Any]:
        """지식의 완성도를 분석합니다."""
        pass

    @abstractmethod
    async def detect_duplicate_nodes(
        self, similarity_threshold: float = 0.9
    ) -> List[tuple[NodeId, NodeId, float]]:
        """중복 노드를 탐지합니다."""
        pass

    @abstractmethod
    async def detect_orphaned_nodes(self) -> List[NodeId]:
        """고립된 노드들을 탐지합니다."""
        pass

    @abstractmethod
    async def analyze_embedding_quality(self) -> Dict[str, Any]:
        """임베딩 품질을 분석합니다."""
        pass

    @abstractmethod
    async def validate_relationship_consistency(self) -> Dict[str, Any]:
        """관계의 일관성을 검증합니다."""
        pass
