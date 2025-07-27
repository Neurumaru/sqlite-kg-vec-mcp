"""
Document 트랜잭션 통합 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock

from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.document_processor import (
    DocumentProcessor,
    KnowledgeExtractionResult,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentTransactionIntegration(unittest.IsolatedAsyncioTestCase):
    """Document 트랜잭션 통합 테스트."""

    def setUp(self):
        """테스트 설정."""
        # Mock 데이터베이스
        self.mock_database = Mock()
        
        # Mock transaction context manager
        from unittest.mock import MagicMock
        
        transaction_mock = MagicMock()
        transaction_mock.__aenter__ = AsyncMock()
        transaction_mock.__aexit__ = AsyncMock(return_value=None)
        self.mock_database.transaction.return_value = transaction_mock
        
        # Mock 지식 추출기
        self.mock_knowledge_extractor = Mock()
        
        # Repository와 Processor 생성
        self.document_repository = SQLiteDocumentRepository(self.mock_database)
        self.document_processor = DocumentProcessor(
            self.mock_knowledge_extractor, self.document_repository
        )

        # 샘플 데이터
        self.sample_document = Document(
            id=DocumentId.generate(),
            title="통합 테스트 문서",
            content="이것은 통합 테스트용 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        self.sample_node = Node(
            id=NodeId.generate(),
            name="테스트 개체",
            node_type=NodeType.CONCEPT,
        )

        self.sample_relationship = Relationship(
            id=RelationshipId.generate(),
            source_node_id=NodeId.generate(),
            target_node_id=NodeId.generate(),
            relationship_type=RelationshipType.CONTAINS,
            label="포함",
        )

    async def test_successful_document_processing_with_persistence(self):
        """영속성을 포함한 문서 처리 성공 시나리오 테스트."""
        # Given: 지식 추출이 성공적으로 수행됨
        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            return_value=([self.sample_node], [self.sample_relationship])
        )

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서가 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)  # 명령 성공

        # When: 문서를 처리
        result = await self.document_processor.process_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(result.nodes[0], self.sample_node)
        self.assertEqual(result.relationships[0], self.sample_relationship)

        # 문서 상태가 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

        # 연결된 요소들이 문서에 추가됨
        self.assertIn(self.sample_node.id, self.sample_document.connected_nodes)
        self.assertIn(self.sample_relationship.id, self.sample_document.connected_relationships)

        # 데이터베이스 호출 검증
        self.mock_database.execute_command.assert_called()  # save 및 update 호출됨

    async def test_failed_document_processing_with_rollback(self):
        """문서 처리 실패 시 롤백 시나리오 테스트."""
        # Given: 지식 추출이 실패함
        error_message = "지식 추출 실패"
        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            side_effect=Exception(error_message)
        )

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서가 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)  # 명령 성공

        # When & Then: 문서 처리가 실패하고 예외가 발생
        with self.assertRaises(Exception) as context:
            await self.document_processor.process_document(self.sample_document)

        self.assertEqual(str(context.exception), error_message)

        # 문서 상태가 FAILED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.FAILED)
        self.assertEqual(self.sample_document.metadata["error"], error_message)

        # 실패 상태 업데이트를 위한 데이터베이스 호출 검증
        self.mock_database.execute_command.assert_called()

    async def test_concurrent_document_processing_conflict(self):
        """동시 문서 처리 충돌 시나리오 테스트."""
        # Given: 두 개의 프로세서가 동일한 문서 처리 시도
        processor1 = DocumentProcessor(
            self.mock_knowledge_extractor, self.document_repository
        )
        processor2 = DocumentProcessor(
            self.mock_knowledge_extractor, self.document_repository
        )

        # 첫 번째 저장은 성공
        # 두 번째 저장 시 문서가 이미 존재함을 시뮬레이션
        existing_doc_row = {
            "id": str(self.sample_document.id),
            "title": self.sample_document.title,
            "content": self.sample_document.content,
            "doc_type": self.sample_document.doc_type.value,
            "status": DocumentStatus.PROCESSING.value,
            "metadata": "{}",
            "version": 1,
            "created_at": self.sample_document.created_at.isoformat(),
            "updated_at": self.sample_document.updated_at.isoformat(),
            "processed_at": None,
            "connected_nodes": "[]",
            "connected_relationships": "[]",
        }

        call_count = 0
        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 첫 번째 문서 처리를 위한 호출들
            if call_count <= 3:  # exists(), save(), update_with_knowledge() 에 대한 조회들
                return []  # 문서 존재하지 않음
            else:
                return [existing_doc_row]  # 두 번째 문서 처리 시 문서 존재함

        self.mock_database.execute_query = AsyncMock(side_effect=mock_query)
        self.mock_database.execute_command = AsyncMock(return_value=1)

        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            return_value=([], [])
        )

        # When: 첫 번째 프로세서는 성공
        document1 = Document(
            id=self.sample_document.id,
            title=self.sample_document.title,
            content=self.sample_document.content,
            doc_type=self.sample_document.doc_type,
        )

        await processor1.process_document(document1)

        # 두 번째 프로세서는 DocumentAlreadyExistsException 발생
        document2 = Document(
            id=self.sample_document.id,
            title=self.sample_document.title,
            content=self.sample_document.content,
            doc_type=self.sample_document.doc_type,
        )

        from src.domain.exceptions.document_exceptions import DocumentAlreadyExistsException, ConcurrentModificationError
        
        # 두 번째 처리 시에는 DocumentAlreadyExistsException 또는 ConcurrentModificationError 발생
        with self.assertRaises((DocumentAlreadyExistsException, ConcurrentModificationError)):
            await processor2.process_document(document2)

    async def test_document_reprocessing_with_persistence(self):
        """영속성을 포함한 문서 재처리 테스트."""
        # Given: 이미 처리된 문서
        self.sample_document.mark_as_processed()
        self.sample_document.add_connected_node(NodeId.generate())
        self.sample_document.add_connected_relationship(RelationshipId.generate())

        # 새로운 지식 추출 결과
        new_node = Node(
            id=NodeId.generate(),
            name="새로운 개체",
            node_type=NodeType.PERSON,
        )

        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            return_value=([new_node], [])
        )

        # Database mocking for update operations
        # 여러 호출에 대해 다른 응답 제공
        call_count = 0
        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # exists() 호출에 대한 응답
            if "SELECT 1 FROM documents WHERE id = ?" in str(args):
                return [{"1": 1}]  # 문서 존재함
            
            # 다른 조회에 대한 응답
            return [{
                "id": str(self.sample_document.id),
                "title": self.sample_document.title,
                "content": self.sample_document.content,
                "doc_type": self.sample_document.doc_type.value,
                "status": self.sample_document.status.value,
                "metadata": "{}",
                "version": self.sample_document.version,  # 현재 문서와 동일한 버전
                "created_at": self.sample_document.created_at.isoformat(),
                "updated_at": self.sample_document.updated_at.isoformat(),
                "processed_at": self.sample_document.processed_at.isoformat() if self.sample_document.processed_at else None,
                "connected_nodes": "[]",
                "connected_relationships": "[]",
            }]

        self.mock_database.execute_query = AsyncMock(side_effect=mock_query)
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 문서 재처리
        result = await self.document_processor.reprocess_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0], new_node)

        # 기존 연결 정보가 초기화되고 새로운 노드만 연결됨
        self.assertEqual(len(self.sample_document.connected_nodes), 1)
        self.assertEqual(self.sample_document.connected_nodes[0], new_node.id)
        self.assertEqual(len(self.sample_document.connected_relationships), 0)

        # 상태가 다시 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

    async def test_batch_document_processing(self):
        """일괄 문서 처리 테스트."""
        # Given: 여러 문서
        documents = [
            Document(
                id=DocumentId.generate(),
                title=f"문서 {i}",
                content=f"내용 {i}",
                doc_type=DocumentType.TEXT,
            )
            for i in range(3)
        ]

        # 각 문서마다 다른 지식 추출 결과
        extraction_results = [
            ([Node(id=NodeId.generate(), name=f"노드 {i}", node_type=NodeType.CONCEPT)], [])
            for i in range(3)
        ]

        call_count = 0
        async def mock_extract_knowledge(doc):
            nonlocal call_count
            result = extraction_results[call_count]
            call_count += 1
            return result

        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            side_effect=mock_extract_knowledge
        )

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서들이 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 모든 문서 처리
        results = []
        for doc in documents:
            result = await self.document_processor.process_document(doc)
            results.append(result)

        # Then: 결과 검증
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(len(result.nodes), 1)
            self.assertEqual(result.nodes[0].name, f"노드 {i}")
            self.assertEqual(documents[i].status, DocumentStatus.PROCESSED)

        # 모든 데이터베이스 호출이 성공적으로 이루어짐
        self.assertEqual(self.mock_database.execute_command.call_count, 6)  # save + update 각 3번

    async def test_document_processing_with_empty_extraction_result(self):
        """빈 지식 추출 결과로 문서 처리 테스트."""
        # Given: 지식 추출 결과가 비어있음
        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            return_value=([], [])
        )

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 문서 처리
        result = await self.document_processor.process_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertTrue(result.is_empty())
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.relationships), 0)

        # 문서 상태는 여전히 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

        # 연결된 요소가 없음
        self.assertEqual(len(self.sample_document.connected_nodes), 0)
        self.assertEqual(len(self.sample_document.connected_relationships), 0)


if __name__ == "__main__":
    unittest.main()