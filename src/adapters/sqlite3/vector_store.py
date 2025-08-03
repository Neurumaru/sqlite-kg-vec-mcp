"""
sqlite-vec 확장을 사용한 VectorStore 포트의 SQLite 구현.
"""

from typing import Any, Optional

from src.config.search_config import SearchConfig
from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.vector import Vector
from src.ports.vector_store import VectorStore

from .vector_reader_impl import SQLiteVectorReader
from .vector_retriever_impl import SQLiteVectorRetriever
from .vector_writer_impl import SQLiteVectorWriter


class SQLiteVectorStore(VectorStore):
    """
    VectorStore 포트의 SQLite 구현.

    이 클래스는 컴포지션 패턴을 사용하여 분리된 구현체들을 통합하여
    완전한 VectorStore 인터페이스를 제공합니다.

    - writer: 데이터 추가/수정/삭제
    - reader: 데이터 조회/검색
    - retriever: 고급 검색/리트리벌
    """

    def __init__(
        self,
        db_path: str,
        table_name: str = "vectors",
        optimize: bool = True,
        search_config: Optional[SearchConfig] = None,
    ):
        """
        SQLite 벡터 저장소 어댑터를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            table_name: 벡터를 저장할 테이블 이름
            optimize: 최적화 PRAGMA 적용 여부
            search_config: 검색 설정 (None인 경우 기본값 사용)
        """
        self.writer = SQLiteVectorWriter(db_path, table_name, optimize)
        self.reader = SQLiteVectorReader(db_path, table_name, optimize)
        self.retriever = SQLiteVectorRetriever(db_path, table_name, optimize, search_config)

    def __getattr__(self, name):
        """속성과 메서드를 위임합니다."""
        # writer, reader, retriever 순서로 확인
        for component in [self.writer, self.reader, self.retriever]:
            if hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # VectorWriter 메서드들 위임
    async def add_documents(self, documents: list[DocumentMetadata], **kwargs: Any) -> list[str]:
        """문서를 벡터 저장소에 추가합니다."""
        return await self.writer.add_documents(documents, **kwargs)

    async def add_vectors(
        self, vectors: list[Vector], documents: list[DocumentMetadata], **kwargs: Any
    ) -> list[str]:
        """벡터와 연관된 문서를 벡터 저장소에 추가합니다."""
        return await self.writer.add_vectors(vectors, documents, **kwargs)

    async def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """지정된 ID들의 벡터/문서를 삭제합니다."""
        return await self.writer.delete(ids, **kwargs)

    async def update_document(
        self,
        document_id: str,
        document: DocumentMetadata,
        vector: Optional[Vector] = None,
        **kwargs: Any,
    ) -> bool:
        """문서의 메타데이터를 업데이트합니다."""
        return await self.writer.update_document(document_id, document, vector, **kwargs)

    # VectorReader 메서드들 위임
    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """문서 ID로 문서 메타데이터를 조회합니다."""
        return await self.reader.get_document(document_id)

    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """벡터 ID로 벡터를 조회합니다."""
        return await self.reader.get_vector(vector_id)

    async def list_documents(
        self, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs: Any
    ) -> list[DocumentMetadata]:
        """저장된 문서 목록을 조회합니다."""
        return await self.reader.list_documents(limit, offset, **kwargs)

    async def count_documents(self, **kwargs: Any) -> int:
        """저장된 문서 수를 반환합니다."""
        return await self.reader.count_documents(**kwargs)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_criteria: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[tuple[DocumentMetadata, float]]:
        """쿼리 텍스트를 기반으로 유사도 검색을 수행합니다."""
        return await self.reader.similarity_search(query, k, filter_criteria, **kwargs)

    async def similarity_search_by_vector(
        self,
        vector: Vector,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[DocumentMetadata, float]]:
        """임베딩 벡터를 기반으로 유사도 검색을 수행합니다."""
        return await self.reader.similarity_search_by_vector(vector, k, **kwargs)

    # VectorRetriever 메서드들 위임
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """쿼리를 기반으로 관련 문서를 검색합니다."""
        return await self.retriever.retrieve(query, k, search_type, **kwargs)

    async def retrieve_with_filter(
        self,
        query: str,
        filter_criteria: dict[str, Any],
        k: int = 4,
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """필터가 적용된 문서 검색을 수행합니다."""
        return await self.retriever.retrieve_with_filter(query, filter_criteria, k, **kwargs)

    async def retrieve_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_criteria: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """MMR을 사용한 다양성 기반 검색을 수행합니다."""
        return await self.retriever.retrieve_mmr(
            query, k, fetch_k, lambda_mult, filter_criteria, **kwargs
        )

    async def get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[DocumentMetadata]:
        """쿼리와 관련된 문서들을 반환합니다."""
        return await self.retriever.get_relevant_documents(query, **kwargs)
