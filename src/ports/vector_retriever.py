"""
벡터 저장소 검색 기능을 위한 포트 인터페이스.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.search_result import VectorSearchResultCollection


class VectorRetriever(ABC):
    """
    벡터 저장소에서 고급 검색 기능을 제공하는 포트.

    인터페이스 분리 원칙(ISP)에 따라 검색/리트리벌 작업만을 담당합니다.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        다양한 검색 타입으로 문서를 검색합니다.

        인자:
            query: 검색 쿼리
            k: 반환할 결과 수
            search_type: 검색 타입 ("similarity", "mmr", "similarity_score_threshold")
            **kwargs: 검색 타입별 추가 옵션

        반환:
            VectorSearchResultCollection 객체
        """

    @abstractmethod
    async def retrieve_with_filter(
        self,
        query: str,
        filter_criteria: dict[str, Any],
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        필터 조건과 함께 문서를 검색합니다.

        인자:
            query: 검색 쿼리
            filter_criteria: 필터 조건
            k: 반환할 결과 수
            **kwargs: 추가 검색 옵션

        반환:
            VectorSearchResultCollection 객체
        """

    @abstractmethod
    async def retrieve_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        MMR(Maximal Marginal Relevance) 알고리즘으로 문서를 검색합니다.

        인자:
            query: 검색 쿼리
            k: 반환할 결과 수
            fetch_k: MMR 계산을 위해 먼저 가져올 문서 수
            lambda_mult: 관련성과 다양성 간의 균형 (0~1)
            **kwargs: 추가 검색 옵션

        반환:
            VectorSearchResultCollection 객체
        """

    @abstractmethod
    async def get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        쿼리와 관련된 문서를 검색합니다 (LangChain 호환).

        인자:
            query: 검색 쿼리
            **kwargs: 검색 옵션

        반환:
            VectorSearchResultCollection 객체
        """
