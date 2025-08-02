"""
SQLiteDocumentRepository 예외 처리 테스트.
"""

# pylint: disable=protected-access

import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.adapters.sqlite3.exceptions import (
    SQLiteConnectionException,
    SQLiteIntegrityException,
    SQLiteOperationalException,
    SQLiteTimeoutException,
    SQLiteTransactionException,
)
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.exceptions.document_exceptions import (
    ConcurrentModificationError,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto.document import (
    DocumentData,
)
from src.dto.document import DocumentStatus as DTODocumentStatus
from src.dto.document import DocumentType as DTODocumentType


class TestDocumentRepositoryExceptions(unittest.IsolatedAsyncioTestCase):
    """DocumentRepository 예외 처리 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_database = AsyncMock()

        # Mock transaction context manager
        @asynccontextmanager
        async def mock_transaction():
            yield None

        self.mock_database.transaction = mock_transaction

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

    async def test_save_database_connection_failure(self):
        """데이터베이스 연결 실패 시 구체적 예외 테스트."""
        # Given
        sample_data = self._document_to_data(self.sample_document)
        # 데이터베이스 연결 오류 시뮬레이션
        connection_error = SQLiteConnectionException(
            "/test/db.sqlite", "database connection failed"
        )
        self.mock_database.execute_query = AsyncMock(side_effect=connection_error)

        # When & Then
        with self.assertRaises(SQLiteConnectionException) as context:
            await self.repository.save(sample_data)

        self.assertEqual(context.exception.db_path, "/test/db.sqlite")
        self.assertEqual(context.exception.error_code, "SQLITE_CONNECTION_FAILED")

    async def test_save_integrity_constraint_violation(self):
        """무결성 제약 위반 시 구체적 예외 테스트."""
        # Given
        sample_data = self._document_to_data(self.sample_document)
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서 없음 확인

        # UNIQUE 제약 위반 시뮬레이션
        integrity_error = SQLiteIntegrityException(
            constraint="UNIQUE", table="documents", column="id", value=str(self.sample_document.id)
        )
        self.mock_database.execute_command = AsyncMock(side_effect=integrity_error)

        # When & Then
        with self.assertRaises(SQLiteIntegrityException) as context:
            await self.repository.save(sample_data)

        self.assertEqual(context.exception.constraint, "UNIQUE")
        self.assertEqual(context.exception.table, "documents")
        self.assertEqual(context.exception.column, "id")
        self.assertEqual(context.exception.value, str(self.sample_document.id))

    async def test_update_operational_error_with_retry_logic(self):
        """운영 오류 발생 시 재시도 로직 테스트."""
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

        # 데이터베이스 잠금 오류 시뮬레이션
        operational_error = SQLiteOperationalException(
            operation="UPDATE", message="database is locked"
        )
        self.mock_database.execute_command = AsyncMock(side_effect=operational_error)

        sample_data = self._document_to_data(self.sample_document)

        # When & Then
        with self.assertRaises(SQLiteOperationalException) as context:
            await self.repository.update(sample_data)

        self.assertEqual(context.exception.operation, "UPDATE")
        self.assertIn("database is locked", context.exception.message)

    async def test_transaction_rollback_on_update_with_knowledge_failure(self):
        """지식 요소 업데이트 중 트랜잭션 롤백 테스트."""
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

        # 트랜잭션 실패 시뮬레이션
        transaction_error = SQLiteTransactionException(
            transaction_id="tx_001", state="COMMITTING", message="transaction deadlock detected"
        )
        self.mock_database.execute_command = AsyncMock(side_effect=transaction_error)

        node_ids = [str(NodeId.generate())]
        relationship_ids = [str(RelationshipId.generate())]
        sample_data = self._document_to_data(self.sample_document)

        # When & Then
        with self.assertRaises(SQLiteTransactionException) as context:
            await self.repository.update_with_knowledge(sample_data, node_ids, relationship_ids)

        self.assertEqual(context.exception.transaction_id, "tx_001")
        self.assertEqual(context.exception.state, "COMMITTING")
        self.assertIn("deadlock", context.exception.message)

    async def test_timeout_during_long_running_query(self):
        """장시간 실행 쿼리 타임아웃 테스트."""
        # Given
        timeout_error = SQLiteTimeoutException(
            operation="SELECT",
            timeout_duration=30.0,
            query="SELECT * FROM documents WHERE status = ?",
        )
        self.mock_database.execute_query = AsyncMock(side_effect=timeout_error)

        # When & Then
        with self.assertRaises(SQLiteTimeoutException) as context:
            await self.repository.find_by_status(DocumentStatus.PENDING)

        self.assertIn("SELECT", context.exception.operation)
        self.assertEqual(context.exception.timeout_duration, 30.0)
        self.assertIn("SELECT * FROM documents", context.exception.query)

    async def test_multiple_exception_types_in_complex_operation(self):
        """복잡한 작업에서 다양한 예외 타입 처리 테스트."""
        # Given - 복잡한 시나리오: exists 체크 -> save -> update
        sample_data = self._document_to_data(self.sample_document)

        # 1단계: exists 체크 시 연결 오류
        connection_error = SQLiteConnectionException("/test/db.sqlite", "connection lost")
        self.mock_database.execute_query = AsyncMock(side_effect=connection_error)

        # When & Then - 첫 번째 예외 (연결 오류)
        with self.assertRaises(SQLiteConnectionException):
            await self.repository.save(sample_data)

        # Given - 2단계: 연결 복구 후 무결성 위반
        self.mock_database.execute_query = AsyncMock(return_value=[])  # exists 성공
        integrity_error = SQLiteIntegrityException("UNIQUE", "documents", "id")
        self.mock_database.execute_command = AsyncMock(side_effect=integrity_error)

        # When & Then - 두 번째 예외 (무결성 위반)
        with self.assertRaises(SQLiteIntegrityException):
            await self.repository.save(sample_data)

    async def test_exception_error_code_and_message_consistency(self):
        """예외의 에러 코드와 메시지 일관성 테스트."""
        # Given
        sample_data = self._document_to_data(self.sample_document)
        self.mock_database.execute_query = AsyncMock(return_value=[])

        integrity_error = SQLiteIntegrityException(
            constraint="FOREIGN_KEY", table="documents", column="parent_id", value="invalid_parent"
        )
        self.mock_database.execute_command = AsyncMock(side_effect=integrity_error)

        # When & Then
        with self.assertRaises(SQLiteIntegrityException) as context:
            await self.repository.save(sample_data)

        # 에러 코드와 메시지 형식 일관성 검증
        self.assertEqual(context.exception.error_code, "SQLITE_INTEGRITY_VIOLATION")
        self.assertIn("FOREIGN_KEY", context.exception.message)
        self.assertIn("documents.parent_id", context.exception.message)
        self.assertIn("invalid_parent", context.exception.message)

        # 문자열 표현 형식 검증
        expected_str_format = f"[{context.exception.error_code}] {context.exception.message}"
        self.assertEqual(str(context.exception), expected_str_format)


if __name__ == "__main__":
    unittest.main()
