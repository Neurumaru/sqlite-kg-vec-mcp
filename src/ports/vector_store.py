"""
벡터 저장소 포트 (LangChain 호환).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


class VectorStore(ABC):
    """
    벡터 저장소 포트 (LangChain 호환).

    LangChain의 VectorStore 인터페이스와 호환되는 벡터 저장 및 검색 기능을 제공합니다.
    """

    # Core LangChain VectorStore 메서드들
    @abstractmethod
    async def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        문서들을 벡터 저장소에 추가합니다 (LangChain 표준).

        Args:
            documents: 추가할 Document 객체들
            **kwargs: 추가 옵션들

        Returns:
            추가된 문서들의 ID 리스트
        """

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """
        유사도 검색을 수행합니다 (LangChain 표준).

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션들 (filter 등)

        Returns:
            유사한 Document 객체들의 리스트
        """

    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        유사도 검색을 수행하고 점수와 함께 반환합니다 (LangChain 표준).

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션들

        Returns:
            (Document, 유사도점수) 튜플들의 리스트
        """

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """
        벡터로 유사도 검색을 수행합니다 (LangChain 표준).

        Args:
            embedding: 쿼리 벡터 (임베딩)
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션들

        Returns:
            유사한 Document 객체들의 리스트
        """

    @abstractmethod
    async def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        문서들을 삭제합니다 (LangChain 표준).

        Args:
            ids: 삭제할 문서 ID들
            **kwargs: 추가 삭제 옵션들

        Returns:
            삭제 성공 여부
        """

    # 클래스 메서드들 (LangChain 패턴)
    @classmethod
    @abstractmethod
    async def from_documents(
        cls,
        documents: List[Document],
        embedding: Any,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        문서들로부터 벡터 저장소를 생성합니다 (LangChain 표준).

        Args:
            documents: 초기 Document 객체들
            embedding: 임베딩 모델
            **kwargs: 추가 생성 옵션들

        Returns:
            생성된 VectorStore 인스턴스
        """

    @classmethod
    @abstractmethod
    async def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        텍스트들로부터 벡터 저장소를 생성합니다 (LangChain 표준).

        Args:
            texts: 텍스트 리스트
            embedding: 임베딩 모델
            metadatas: 각 텍스트의 메타데이터들
            **kwargs: 추가 생성 옵션들

        Returns:
            생성된 VectorStore 인스턴스
        """

    # 추가 헬퍼 메서드들
    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """
        벡터 저장소를 Retriever로 변환합니다 (LangChain 표준).

        Args:
            **kwargs: Retriever 설정 옵션들

        Returns:
            Retriever 인스턴스
        """
