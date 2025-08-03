"""
문서 메타데이터를 나타내는 도메인 값 객체.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DocumentMetadata:
    """
    문서의 메타데이터를 나타내는 불변 값 객체.

    LangChain Document의 순수한 도메인 대체재입니다.
    """

    content: str
    metadata: dict[str, Any]
    source: Optional[str] = None

    def __post_init__(self):
        """유효성 검사."""
        if not isinstance(self.content, str):
            raise ValueError("content는 문자열이어야 합니다")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata는 딕셔너리여야 합니다")
        if self.source is not None and not isinstance(self.source, str):
            raise ValueError("source는 문자열이거나 None이어야 합니다")

    @classmethod
    def create(
        cls, content: str, metadata: Optional[dict[str, Any]] = None, source: Optional[str] = None
    ) -> "DocumentMetadata":
        """편의 팩토리 메서드."""
        return cls(content=content, metadata=metadata or {}, source=source)

    def with_metadata(self, **kwargs: Any) -> "DocumentMetadata":
        """새로운 메타데이터로 새 인스턴스 생성."""
        new_metadata = {**self.metadata, **kwargs}
        return DocumentMetadata(content=self.content, metadata=new_metadata, source=self.source)
