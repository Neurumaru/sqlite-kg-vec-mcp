"""
문서 영속성 도메인 서비스.
"""

from typing import Optional

from src.common.observability.logger import ObservableLogger
from src.domain.entities.document import Document
from src.ports.mappers import DocumentMapper
from src.ports.repositories.document import DocumentRepository


class DocumentPersistenceService:
    """
    문서 영속성 관리 도메인 서비스.

    문서의 저장, 업데이트, 상태 관리를 담당합니다.
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_mapper: DocumentMapper,
        logger: Optional[ObservableLogger] = None,
    ):
        self.document_repository = document_repository
        self.document_mapper = document_mapper
        from src.common.observability.logger import get_logger

        self.logger = logger or get_logger("document_persistence", "domain")

    async def save_or_update_document(self, document: Document) -> None:
        """문서를 저장하거나 업데이트합니다."""
        document_data = self.document_mapper.to_data(document)
        if await self.document_repository.exists(str(document.id)):
            await self.document_repository.update(document_data)
            self.logger.info("document_updated", document_id=str(document.id))
        else:
            await self.document_repository.save(document_data)
            self.logger.info("document_saved", document_id=str(document.id))

    async def update_document_with_knowledge(
        self, document: Document, node_ids: list[str], relationship_ids: list[str]
    ) -> None:
        """문서를 지식 요소들과 함께 업데이트합니다."""
        document_data = self.document_mapper.to_data(document)
        await self.document_repository.update_with_knowledge(
            document_data, node_ids, relationship_ids
        )
        self.logger.info(
            "document_knowledge_updated",
            document_id=str(document.id),
            node_count=len(node_ids),
            relationship_count=len(relationship_ids),
        )

    async def update_document_status(self, document: Document) -> None:
        """문서 상태만 업데이트합니다."""
        document_data = self.document_mapper.to_data(document)
        await self.document_repository.update(document_data)
        self.logger.info(
            "document_status_updated", document_id=str(document.id), status=document.status.value
        )
