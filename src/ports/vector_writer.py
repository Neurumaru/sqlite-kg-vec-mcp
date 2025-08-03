"""
벡터 저장소 쓰기 작업을 위한 포트 인터페이스.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.vector import Vector


class VectorWriter(ABC):
    """
    벡터 저장소에 데이터를 추가/삭제하는 기능을 위한 포트.

    인터페이스 분리 원칙(ISP)에 따라 쓰기 작업만을 담당합니다.
    """

    @abstractmethod
    async def add_documents(self, documents: list[DocumentMetadata], **kwargs: Any) -> list[str]:
        """
        문서를 벡터 저장소에 추가합니다.

        인자:
            documents: 추가할 DocumentMetadata 객체 목록
            **kwargs: 추가 옵션

        반환:
            추가된 문서 ID 목록
        """

    @abstractmethod
    async def add_vectors(
        self, vectors: list[Vector], documents: list[DocumentMetadata], **kwargs: Any
    ) -> list[str]:
        """
        벡터와 문서를 함께 저장소에 추가합니다.

        인자:
            vectors: 추가할 Vector 객체 목록
            documents: 벡터에 대응하는 DocumentMetadata 객체 목록
            **kwargs: 추가 옵션

        반환:
            추가된 문서 ID 목록
        """

    @abstractmethod
    async def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        문서를 삭제합니다.

        인자:
            ids: 삭제할 문서 ID 목록
            **kwargs: 추가 삭제 옵션

        반환:
            삭제 성공 여부
        """

    @abstractmethod
    async def update_document(
        self,
        document_id: str,
        document: DocumentMetadata,
        vector: Optional[Vector] = None,
        **kwargs: Any,
    ) -> bool:
        """
        문서와 벡터를 업데이트합니다.

        인자:
            document_id: 업데이트할 문서 ID
            document: 새로운 DocumentMetadata
            vector: 새로운 Vector (선택사항)
            **kwargs: 추가 업데이트 옵션

        반환:
            업데이트 성공 여부
        """
