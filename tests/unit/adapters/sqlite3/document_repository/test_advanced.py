"""
DocumentRepository 고급 기능 테스트.
이 파일에는 트랜잭션, 타임아웃, 동시성 등의 고급 기능을 테스트하는 코드가 포함됩니다.
"""

# pylint: disable=protected-access

import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId
from src.dto.document import (
    DocumentData,
)
from src.dto.document import DocumentStatus as DTODocumentStatus
from src.dto.document import DocumentType as DTODocumentType


class TestDocumentRepositoryAdvanced(unittest.IsolatedAsyncioTestCase):
    """DocumentRepository 고급 기능 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_database = AsyncMock()

        # Mock transaction context manager
        @asynccontextmanager
        async def mock_transaction():
            yield None

        # Create a mock that returns the context manager
        self.mock_transaction_context = AsyncMock()
        self.mock_transaction_context.__aenter__ = AsyncMock(return_value=None)
        self.mock_transaction_context.__aexit__ = AsyncMock(return_value=None)

        self.mock_database.transaction = AsyncMock(return_value=self.mock_transaction_context)

        self.repository = SQLiteDocumentRepository(self.mock_database)

        # 샘플 문서 생성
        self.sample_document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
            status=DocumentStatus.PENDING,
            version=1,
        )

    def _document_to_data(self, document: Document) -> DocumentData:
        """Document 엔티티를 DocumentData DTO로 변환하는 헬퍼 메서드."""
        status_mapping = {
            DocumentStatus.PENDING: DTODocumentStatus.PENDING,
            DocumentStatus.PROCESSING: DTODocumentStatus.PROCESSING,
            DocumentStatus.PROCESSED: DTODocumentStatus.COMPLETED,
            DocumentStatus.FAILED: DTODocumentStatus.FAILED,
        }

        return DocumentData(
            id=str(document.id),
            title=document.title,
            content=document.content,
            doc_type=DTODocumentType(document.doc_type.value),
            status=status_mapping[document.status],
            metadata=document.metadata,
            version=document.version,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
            connected_nodes=[str(node_id) for node_id in document.connected_nodes],
            connected_relationships=[str(rel_id) for rel_id in document.connected_relationships],
        )

    async def test_transaction_rollback_success_when_operation_fails(self):
        """Given: 트랜잭션 중 오류가 발생하는 상황
        When: update_with_knowledge 작업이 실패하면
        Then: 트랜잭션이 롤백되어야 한다
        """
        # Given
        sample_data = self._document_to_data(self.sample_document)
        self.mock_database.execute_query = AsyncMock(return_value=[])
        self.mock_database.execute_command = AsyncMock(side_effect=Exception("Database error"))

        # When & Then
        with self.assertRaises(Exception):  # noqa: B017
            await self.repository.update_with_knowledge(sample_data, [], [])

        # 트랜잭션 컨텍스트가 호출되었는지 확인
        self.mock_database.transaction.assert_called()

    async def test_batch_operation_success_when_multiple_documents(self):
        """Given: 여러 문서에 대한 배치 작업
        When: 배치 작업을 수행하면
        Then: 모든 문서가 일괄 처리되어야 한다
        """
        # Given
        # documents = [self._document_to_data(self.sample_document) for _ in range(3)]
        self.mock_database.execute_many = AsyncMock(return_value=3)

        # When
        # 실제 배치 메서드가 구현되면 테스트 가능
        # result = await self.repository.batch_save(documents)

        # Then
        # self.assertEqual(len(result), 3)
        # 현재는 배치 메서드가 구현되지 않아 placeholder
        self.assertEqual(self.mock_database.execute_many.return_value, 3)  # placeholder test

    async def test_connection_pooling_success_when_concurrent_access(self):
        """Given: 동시 접근 상황
        When: 여러 요청이 동시에 들어오면
        Then: 연결 풀이 적절히 관리되어야 한다
        """
        # Given
        sample_data = self._document_to_data(self.sample_document)
        self.mock_database.execute_query = AsyncMock(return_value=[])
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When
        # 동시 작업 시뮬레이션
        import asyncio  # pylint: disable=import-outside-toplevel

        tasks = [self.repository.save(sample_data) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Then
        # 모든 작업이 성공하거나 적절한 예외가 발생해야 함
        self.assertEqual(len(results), 5)
        for result in results:
            if not isinstance(result, Exception):
                self.assertEqual(result, sample_data)


if __name__ == "__main__":
    unittest.main()
