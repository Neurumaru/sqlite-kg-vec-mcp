"""
문서 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.document import Document
from src.domain.services.document_processor import KnowledgeExtractionResult
from src.domain.value_objects.document_id import DocumentId


class DocumentManagementUseCase(ABC):
    """문서 관리 Use Case 인터페이스."""

    @abstractmethod
    async def create_document(
        self, title: str, content: str, metadata: Optional[dict] = None
    ) -> Document:
        """새 문서를 생성합니다."""
        pass

    @abstractmethod
    async def get_document(self, document_id: DocumentId) -> Optional[Document]:
        """문서를 조회합니다."""
        pass

    @abstractmethod
    async def list_documents(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Document]:
        """문서 목록을 조회합니다."""
        pass

    @abstractmethod
    async def update_document(
        self, document_id: DocumentId, title: Optional[str] = None, 
        content: Optional[str] = None, metadata: Optional[dict] = None
    ) -> Document:
        """문서를 업데이트합니다."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: DocumentId) -> bool:
        """문서를 삭제합니다."""
        pass


class DocumentProcessingUseCase(ABC):
    """문서 처리 Use Case 인터페이스."""

    @abstractmethod
    async def process_document(self, document_id: DocumentId) -> KnowledgeExtractionResult:
        """문서를 처리하여 지식을 추출합니다."""
        pass

    @abstractmethod
    async def reprocess_document(self, document_id: DocumentId) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""
        pass

    @abstractmethod
    async def get_processing_status(self, document_id: DocumentId) -> str:
        """문서 처리 상태를 조회합니다."""
        pass

    @abstractmethod
    async def validate_document_for_processing(self, document_id: DocumentId) -> bool:
        """문서가 처리 가능한 상태인지 검증합니다."""
        pass