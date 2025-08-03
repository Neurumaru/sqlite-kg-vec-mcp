"""
벡터 저장소 읽기 작업을 위한 포트 인터페이스.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.search_result import VectorSearchResultCollection
from src.domain.value_objects.vector import Vector


class VectorReader(ABC):
    """
    벡터 저장소에서 데이터를 조회하는 기능을 위한 포트.

    인터페이스 분리 원칙(ISP)에 따라 읽기 작업만을 담당합니다.
    """

    @abstractmethod
    async def get_document(self, document_id: str) -> DocumentMetadata | None:
        """
        ID로 문서를 조회합니다.

        인자:
            document_id: 조회할 문서 ID

        반환:
            DocumentMetadata 객체 또는 None
        """

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Vector | None:
        """
        ID로 벡터를 조회합니다.

        인자:
            vector_id: 조회할 벡터 ID

        반환:
            Vector 객체 또는 None
        """

    @abstractmethod
    async def list_documents(
        self, limit: int | None = None, offset: int | None = None, **kwargs: Any
    ) -> list[DocumentMetadata]:
        """
        저장된 문서 목록을 조회합니다.

        인자:
            limit: 반환할 문서 수 제한
            offset: 시작 오프셋
            **kwargs: 추가 필터링 옵션

        반환:
            DocumentMetadata 객체 목록
        """

    @abstractmethod
    async def count_documents(self, **kwargs: Any) -> int:
        """
        저장된 문서 수를 반환합니다.

        인자:
            **kwargs: 필터링 옵션

        반환:
            문서 수
        """

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        텍스트 쿼리로 유사도 검색을 수행합니다.

        인자:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션 (필터 등)

        반환:
            VectorSearchResultCollection 객체
        """

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        vector: Vector,
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        벡터로 유사도 검색을 수행합니다.

        인자:
            vector: 쿼리 Vector 객체
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션

        반환:
            VectorSearchResultCollection 객체
        """
