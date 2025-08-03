"""
벡터 검색 결과를 나타내는 도메인 값 객체.
"""

from dataclasses import dataclass
from typing import Any

from .document_metadata import DocumentMetadata


@dataclass(frozen=True)
class VectorSearchResult:
    """
    벡터 검색 결과를 나타내는 불변 값 객체.

    LangChain의 (Document, score) 튜플의 순수한 도메인 대체재입니다.
    """

    document: DocumentMetadata
    score: float
    id: str | None = None

    def __post_init__(self):
        """유효성 검사."""
        if not isinstance(self.document, DocumentMetadata):
            raise ValueError("document는 DocumentMetadata 인스턴스여야 합니다")
        if not isinstance(self.score, int | float):
            raise ValueError("score는 숫자여야 합니다")
        if self.score < 0.0 or self.score > 1.0:
            raise ValueError("score는 0.0과 1.0 사이여야 합니다")
        if self.id is not None and not isinstance(self.id, str):
            raise ValueError("id는 문자열이거나 None이어야 합니다")

    @classmethod
    def create(
        cls,
        content: str,
        score: float,
        metadata: dict[str, Any] | None = None,
        result_id: str | None = None,
        source: str | None = None,
    ) -> "VectorSearchResult":
        """편의 팩토리 메서드."""
        document = DocumentMetadata.create(content=content, metadata=metadata, source=source)
        return cls(document=document, score=score, id=result_id)


@dataclass(frozen=True)
class VectorSearchResultCollection:
    """
    벡터 검색 결과 컬렉션을 나타내는 불변 값 객체.
    """

    results: list[VectorSearchResult]
    total_count: int
    query: str

    def __post_init__(self):
        """유효성 검사."""
        if not isinstance(self.results, list):
            raise ValueError("results는 리스트여야 합니다")
        if not all(isinstance(r, VectorSearchResult) for r in self.results):
            raise ValueError("모든 결과는 VectorSearchResult 인스턴스여야 합니다")
        if not isinstance(self.total_count, int) or self.total_count < 0:
            raise ValueError("total_count는 음이 아닌 정수여야 합니다")
        if not isinstance(self.query, str):
            raise ValueError("query는 문자열이어야 합니다")

    def top_k(self, k: int) -> "VectorSearchResultCollection":
        """상위 k개 결과만 포함하는 새 컬렉션 반환."""
        if k < 0:
            raise ValueError("k는 음이 아닌 정수여야 합니다")

        top_results = self.results[:k]
        return VectorSearchResultCollection(
            results=top_results, total_count=len(top_results), query=self.query
        )

    def filter_by_score(self, min_score: float) -> "VectorSearchResultCollection":
        """최소 점수 이상의 결과만 포함하는 새 컬렉션 반환."""
        filtered_results = [r for r in self.results if r.score >= min_score]
        return VectorSearchResultCollection(
            results=filtered_results, total_count=len(filtered_results), query=self.query
        )
