"""
문서 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.domain.entities.document import Document
from src.domain.services.document_processor import KnowledgeExtractionResult
from src.domain.value_objects.document_id import DocumentId


class DocumentManagementUseCase(ABC):
    """문서 관리 Use Case 인터페이스."""

    @abstractmethod
    async def create_document(
        self, title: str, content: str, metadata: Optional[Dict] = None
    ) -> Document:
        """새 문서를 생성합니다.

        Args:
            title: 문서 제목 (비어있지 않아야 함)
            content: 문서 내용 (비어있지 않아야 함)
            metadata: 추가 메타데이터

        Returns:
            생성된 Document 객체

        Raises:
            ValueError: title 또는 content가 비어있는 경우
        """

    @abstractmethod
    async def get_document(self, document_id: DocumentId) -> Optional[Document]:
        """문서를 조회합니다."""

    @abstractmethod
    async def list_documents(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Document]:
        """문서 목록을 조회합니다."""

    @abstractmethod
    async def update_document(
        self,
        document_id: DocumentId,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """문서를 업데이트합니다."""

    @abstractmethod
    async def delete_document(self, document_id: DocumentId) -> bool:
        """문서를 삭제합니다."""


class DocumentProcessingUseCase(ABC):
    """문서 처리 Use Case 인터페이스."""

    @abstractmethod
    async def process_document(self, document_id: DocumentId) -> KnowledgeExtractionResult:
        """문서를 처리하여 지식을 추출합니다."""

    @abstractmethod
    async def reprocess_document(self, document_id: DocumentId) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""

    @abstractmethod
    async def get_processing_status(self, document_id: DocumentId) -> str:
        """문서 처리 상태를 조회합니다."""

    @abstractmethod
    async def validate_document_for_processing(self, document_id: DocumentId) -> bool:
        """문서가 처리 가능한 상태인지 검증합니다.

        Args:
            document_id: 검증할 문서 ID

        Returns:
            처리 가능 여부

        Raises:
            ValueError: document_id가 유효하지 않은 경우
            DocumentNotFoundException: 문서를 찾을 수 없는 경우
        """

    @abstractmethod
    async def batch_process_documents(
        self, document_ids: List[DocumentId], max_concurrent: int = 5
    ) -> Dict[DocumentId, KnowledgeExtractionResult]:
        """여러 문서를 일괄 처리합니다.

        Args:
            document_ids: 처리할 문서 ID 목록
            max_concurrent: 최대 동시 처리 수

        Returns:
            문서 ID별 처리 결과 딕셔너리

        Raises:
            ValueError: document_ids가 비어있거나 max_concurrent가 잘못된 경우
        """
