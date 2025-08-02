"""
LangChain 호환성을 갖춘 벡터 저장소 포트.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document


class VectorStore(ABC):
    """
    LangChain 호환성을 갖춘 벡터 저장소 포트.

    LangChain의 VectorStore 인터페이스와 호환되는 벡터 저장 및 검색 기능을 제공합니다.
    """

    # 핵심 LangChain VectorStore 메서드
    @abstractmethod
    async def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """
        문서를 벡터 저장소에 추가합니다 (LangChain 표준).

        인자:
            documents: 추가할 Document 객체
            **kwargs: 추가 옵션

        반환:
            추가된 문서 ID 목록
        """

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """
        유사도 검색을 수행합니다 (LangChain 표준).

        인자:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션 (필터 등)

        반환:
            유사한 Document 객체 목록
        """

    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        유사도 검색을 수행하고 점수와 함께 반환합니다 (LangChain 표준).

        인자:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션

        반환:
            (Document, 유사도 점수) 튜플 목록
        """

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """
        벡터로 유사도 검색을 수행합니다 (LangChain 표준).

        인자:
            embedding: 쿼리 벡터 (임베딩)
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션

        반환:
            유사한 Document 객체 목록
        """

    @abstractmethod
    async def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        문서를 삭제합니다 (LangChain 표준).

        인자:
            ids: 삭제할 문서 ID
            **kwargs: 추가 삭제 옵션

        반환:
            삭제 성공 여부
        """

    # 클래스 메서드 (LangChain 패턴)
    @classmethod
    @abstractmethod
    async def from_documents(
        cls,
        documents: list[Document],
        embedding: Any,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        문서로부터 벡터 저장소를 생성합니다 (LangChain 표준).

        인자:
            documents: 초기 Document 객체
            embedding: 임베딩 모델
            **kwargs: 추가 생성 옵션

        반환:
            생성된 VectorStore 인스턴스
        """

    @classmethod
    @abstractmethod
    async def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        텍스트로부터 벡터 저장소를 생성합니다 (LangChain 표준).

        인자:
            texts: 텍스트 목록
            embedding: 임베딩 모델
            metadatas: 각 텍스트에 대한 메타데이터
            **kwargs: 추가 생성 옵션

        반환:
            생성된 VectorStore 인스턴스
        """

    # 추가 도우미 메서드
    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """
        벡터 저장소를 Retriever로 변환합니다 (LangChain 표준).

        인자:
            **kwargs: Retriever 구성 옵션

        반환:
            Retriever 인스턴스
        """
