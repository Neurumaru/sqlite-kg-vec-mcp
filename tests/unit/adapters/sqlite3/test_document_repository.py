"""
SQLiteDocumentRepository 단위 테스트.
"""

# pylint: disable=protected-access

import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.exceptions.document_exceptions import (
    ConcurrentModificationError,
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto.document import (
    DocumentData,
)
from src.dto.document import DocumentStatus as DTODocumentStatus
from src.dto.document import DocumentType as DTODocumentType


class TestSQLiteDocumentRepository(unittest.IsolatedAsyncioTestCase):
    """SQLiteDocumentRepository 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_database = AsyncMock()

        # Mock transaction context manager
        transaction_mock = MagicMock()
        transaction_mock.__aenter__ = AsyncMock()
        transaction_mock.__aexit__ = AsyncMock(return_value=None)
        self.mock_database.transaction.return_value = transaction_mock

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

    async def test_save_success(self):
        """문서 저장 성공 테스트."""
        # Given
        sample_data = self._document_to_data(self.sample_document)
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 기존 문서 없음
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When
        result = await self.repository.save(sample_data)

        # Then
        self.assertEqual(result, sample_data)
        self.mock_database.execute_query.assert_called_once()
        self.mock_database.execute_command.assert_called_once()

    async def test_save_document_already_exists(self):
        """이미 존재하는 문서 저장 시 예외 발생 테스트."""
        # Given
        sample_data = self._document_to_data(self.sample_document)
        existing_row = {
            "id": str(self.sample_document.id),
            "title": "기존 문서",
            "content": "기존 내용",
            "doc_type": "text",
            "status": "pending",
            "metadata": "{}",
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[existing_row])

        # When & Then
        with self.assertRaises(DocumentAlreadyExistsException):
            await self.repository.save(sample_data)

    async def test_find_by_id_success(self):
        """ID로 문서 찾기 성공 테스트."""
        # Given
        row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": self.sample_document.version,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[row])

        # When
        result = await self.repository.find_by_id(str(self.sample_document.id))

        # Then
        self.assertIsNotNone(result)
        self.assertEqual(result.id, str(self.sample_document.id))
        self.assertEqual(result.title, self.sample_document.title)
        self.assertEqual(result.content, self.sample_document.content)

    async def test_find_by_id_not_found(self):
        """ID로 문서 찾기 실패 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[])

        # When
        result = await self.repository.find_by_id(str(self.sample_document.id))

        # Then
        self.assertIsNone(result)

    async def test_find_by_status_success(self):
        """상태로 문서 찾기 성공 테스트."""
        # Given
        row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": self.sample_document.version,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[row])

        # When
        result = await self.repository.find_by_status(DocumentStatus.PENDING)

        # Then
        self.assertEqual(len(result), 1)
        # DocumentData DTO is returned, so we compare the value
        self.assertEqual(result[0].status, DTODocumentStatus.PENDING)

    async def test_update_success(self):
        """문서 업데이트 성공 테스트."""
        # Given
        current_row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": 1,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[current_row])
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # Update document
        sample_data = self._document_to_data(self.sample_document)
        sample_data.title = "업데이트된 제목"

        # When
        result = await self.repository.update(sample_data)

        # Then
        self.assertEqual(result.title, "업데이트된 제목")
        self.assertEqual(result.version, 2)  # 버전이 증가해야 함
        self.mock_database.execute_command.assert_called_once()

    async def test_update_document_not_found(self):
        """존재하지 않는 문서 업데이트 시 예외 발생 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[])

        # When & Then
        sample_data = self._document_to_data(self.sample_document)
        with self.assertRaises(DocumentNotFoundException):
            await self.repository.update(sample_data)

    async def test_update_concurrent_modification_error(self):
        """동시 수정 충돌 시 예외 발생 테스트."""
        # Given
        current_row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": 2,  # 다른 버전
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[current_row])

        # 문서 버전은 1이지만 DB의 버전은 2
        sample_data = self._document_to_data(self.sample_document)
        sample_data.version = 1

        # When & Then
        with self.assertRaises(ConcurrentModificationError) as context:
            await self.repository.update(sample_data)

        self.assertEqual(context.exception.expected_version, 1)
        self.assertEqual(context.exception.actual_version, 2)

    async def test_update_with_knowledge_success(self):
        """지식 요소와 함께 문서 업데이트 성공 테스트."""
        # Given
        current_row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": 1,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[current_row])
        self.mock_database.execute_command = AsyncMock(return_value=1)

        node_ids = [str(NodeId.generate()), str(NodeId.generate())]
        relationship_ids = [str(RelationshipId.generate())]
        sample_data = self._document_to_data(self.sample_document)

        # When
        result = await self.repository.update_with_knowledge(
            sample_data, node_ids, relationship_ids
        )

        # Then
        self.assertEqual(result.connected_nodes, node_ids)
        self.assertEqual(result.connected_relationships, relationship_ids)
        self.assertEqual(result.version, 2)

    async def test_delete_success(self):
        """문서 삭제 성공 테스트."""
        # Given
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When
        result = await self.repository.delete(self.sample_document.id)

        # Then
        self.assertTrue(result)
        self.mock_database.execute_command.assert_called_once()

    async def test_delete_not_found(self):
        """존재하지 않는 문서 삭제 테스트."""
        # Given
        self.mock_database.execute_command = AsyncMock(return_value=0)

        # When
        result = await self.repository.delete(self.sample_document.id)

        # Then
        self.assertFalse(result)

    async def test_exists_true(self):
        """문서 존재 확인 - 존재하는 경우 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[{"1": 1}])

        # When
        result = await self.repository.exists(self.sample_document.id)

        # Then
        self.assertTrue(result)

    async def test_exists_false(self):
        """문서 존재 확인 - 존재하지 않는 경우 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[])

        # When
        result = await self.repository.exists(self.sample_document.id)

        # Then
        self.assertFalse(result)

    async def test_count_by_status(self):
        """상태별 문서 개수 조회 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[{"count": 5}])

        # When
        result = await self.repository.count_by_status(DocumentStatus.PENDING)

        # Then
        self.assertEqual(result, 5)

    async def test_count_total(self):
        """전체 문서 개수 조회 테스트."""
        # Given
        self.mock_database.execute_query = AsyncMock(return_value=[{"count": 10}])

        # When
        result = await self.repository.count_total()

        # Then
        self.assertEqual(result, 10)

    async def test_find_unprocessed(self):
        """미처리 문서 조회 테스트."""
        # Given
        row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": DocumentStatus.PENDING.value,
            "metadata": "{}",
            "version": 1,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[row])

        # When
        result = await self.repository.find_unprocessed(limit=50)

        # Then
        self.assertEqual(len(result), 1)
        # DocumentData DTO is returned, so we compare the value
        self.assertEqual(result[0].status, DTODocumentStatus.PENDING)

    async def test_bulk_update_status(self):
        """상태 일괄 업데이트 테스트."""
        # Given
        document_ids = [DocumentId.generate(), DocumentId.generate()]
        self.mock_database.execute_command = AsyncMock(return_value=2)

        # When
        result = await self.repository.bulk_update_status(document_ids, DocumentStatus.PROCESSED)

        # Then
        self.assertEqual(result, 2)
        self.mock_database.execute_command.assert_called_once()

    async def test_search_content(self):
        """내용 검색 테스트."""
        # Given
        row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": self.sample_document.status.value,
            "metadata": "{}",
            "version": 1,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }
        self.mock_database.execute_query = AsyncMock(return_value=[row])

        # When
        result = await self.repository.search_content("테스트", limit=5)

        # Then
        self.assertEqual(len(result), 1)
        self.assertIn("테스트", result[0].title)

    def test_row_to_document_conversion(self):
        """데이터베이스 행을 Document 엔티티로 변환하는 테스트."""
        # Given
        node_id = NodeId.generate()
        relationship_id = RelationshipId.generate()

        row = {
            "id": str(self.sample_document.id),
            "title": "테스트 문서",
            "content": "테스트 내용",
            "doc_type": "text",
            "status": "pending",
            "metadata": '{"key": "value"}',
            "version": 2,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "processed_at": None,
            "connected_nodes": f'["{node_id}"]',
            "connected_relationships": f'["{relationship_id}"]',
        }

        # When
        document = self.repository._row_to_document(row)

        # Then
        self.assertEqual(document.id, DocumentId(row["id"]))
        self.assertEqual(document.title, "테스트 문서")
        self.assertEqual(document.content, "테스트 내용")
        self.assertEqual(document.doc_type, DocumentType.TEXT)
        self.assertEqual(document.status, DocumentStatus.PENDING)
        self.assertEqual(document.metadata["key"], "value")
        self.assertEqual(document.version, 2)
        self.assertEqual(len(document.connected_nodes), 1)
        self.assertEqual(len(document.connected_relationships), 1)
        self.assertEqual(document.connected_nodes[0], node_id)
        self.assertEqual(document.connected_relationships[0], relationship_id)


if __name__ == "__main__":
    unittest.main()
