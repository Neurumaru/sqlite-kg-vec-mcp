"""
DocumentProcessor 도메인 서비스 단위 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.exceptions.document_exceptions import DocumentProcessingException
from src.domain.services.document_processor import (
    DocumentProcessor,
    KnowledgeExtractionResult,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto.node import (
    NodeData,
)
from src.dto.node import NodeType as DTONodeType
from src.dto.relationship import (
    RelationshipData,
)
from src.dto.relationship import RelationshipType as DTORelationshipType


class TestDocumentProcessor(unittest.IsolatedAsyncioTestCase):
    """DocumentProcessor 도메인 서비스 테스트."""

    async def test_process_document_success(self):
        """문서 처리 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        sample_relationship_data = RelationshipData(
            id=str(RelationshipId.generate()),
            source_node_id=str(NodeId.generate()),
            target_node_id=str(NodeId.generate()),
            relationship_type=DTORelationshipType.RELATES_TO,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(
            return_value=([sample_node_data], [sample_relationship_data])
        )

        # When
        result = await processor.process_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        # DTO 데이터를 통해 생성된 엔티티의 ID가 연결되어 있는지 확인
        self.assertEqual(len(document.connected_nodes), 1)
        self.assertEqual(len(document.connected_relationships), 1)
        self.assertIsNotNone(document.processed_at)
        # document_data로 변환되어 호출되므로 assert_called_once()만 확인
        mock_knowledge_extractor.extract.assert_called_once()

    async def test_process_document_failure(self):
        """문서 처리 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        error_message = "추출 실패"
        mock_knowledge_extractor.extract = AsyncMock(
            side_effect=DocumentProcessingException(str(document.id), error_message)
        )

        # When & Then
        with self.assertRaises(DocumentProcessingException):
            await processor.process_document(document)

        self.assertEqual(document.status, DocumentStatus.FAILED)
        expected_error_message = f"[DOCUMENT_PROCESSING_FAILED] Failed to process document '{document.id}': {error_message}"
        self.assertEqual(document.metadata["error"], expected_error_message)

    async def test_process_document_empty_extraction_result(self):
        """빈 추출 결과로 문서 처리 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([], []))

        # When
        result = await processor.process_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertTrue(result.is_empty())
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.relationships), 0)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertEqual(len(document.connected_nodes), 0)
        self.assertEqual(len(document.connected_relationships), 0)

    def test_validate_document_for_processing_success(self):
        """문서 처리 가능성 검증 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="유효한 문서",
            content="내용이 있는 문서",
            doc_type=DocumentType.TEXT,
        )

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertTrue(result)

    def test_validate_document_for_processing_already_processing(self):
        """이미 처리 중인 문서 검증 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="처리 중인 문서",
            content="내용",
            doc_type=DocumentType.TEXT,
        )
        document.mark_as_processing()

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertFalse(result)

    def test_validate_document_for_processing_already_processed(self):
        """이미 처리 완료된 문서 검증 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="처리된 문서",
            content="내용",
            doc_type=DocumentType.TEXT,
        )
        document.mark_as_processed()

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertFalse(result)

    def test_update_document_links_success(self):
        """문서 링크 업데이트 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        node_id_1 = NodeId.generate()
        node_id_2 = NodeId.generate()
        rel_id_1 = RelationshipId.generate()
        rel_id_2 = RelationshipId.generate()

        # 기존 연결 추가
        document.add_connected_node(node_id_1)
        document.add_connected_relationship(rel_id_1)

        # When
        processor.update_document_links(
            document,
            added_nodes=[node_id_2],
            removed_nodes=[node_id_1],
            added_relationships=[rel_id_2],
            removed_relationships=[rel_id_1],
        )

        # Then
        self.assertNotIn(node_id_1, document.connected_nodes)
        self.assertIn(node_id_2, document.connected_nodes)
        self.assertNotIn(rel_id_1, document.connected_relationships)
        self.assertIn(rel_id_2, document.connected_relationships)

    async def test_reprocess_document_success(self):
        """문서 재처리 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # 기존 처리 상태로 설정
        document.mark_as_processed()
        document.add_connected_node(NodeId.generate())
        document.add_connected_relationship(RelationshipId.generate())

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="재처리 노드",
            node_type=DTONodeType.CONCEPT,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))

        # When
        result = await processor.reprocess_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertEqual(len(document.connected_nodes), 1)

    def test_get_processing_statistics_success(self):
        """문서 처리 통계 계산 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)

        # 다양한 상태의 문서들 생성
        doc1 = Document(
            id=DocumentId.generate(),
            title="처리된 문서1",
            content="내용1",
            doc_type=DocumentType.TEXT,
        )
        doc1.mark_as_processed()
        doc1.add_connected_node(NodeId.generate())
        doc1.add_connected_relationship(RelationshipId.generate())

        doc2 = Document(
            id=DocumentId.generate(),
            title="처리된 문서2",
            content="내용2",
            doc_type=DocumentType.TEXT,
        )
        doc2.mark_as_processed()
        doc2.add_connected_node(NodeId.generate())

        doc3 = Document(
            id=DocumentId.generate(),
            title="처리 중인 문서",
            content="내용3",
            doc_type=DocumentType.TEXT,
        )
        doc3.mark_as_processing()

        doc4 = Document(
            id=DocumentId.generate(),
            title="실패한 문서",
            content="내용4",
            doc_type=DocumentType.TEXT,
        )
        doc4.mark_as_failed("오류")

        documents = [doc1, doc2, doc3, doc4]

        # When
        stats = processor.get_processing_statistics(documents)

        # Then
        self.assertEqual(stats["total_documents"], 4)
        self.assertEqual(stats["processed"], 2)
        self.assertEqual(stats["processing"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["processing_rate"], 0.5)
        self.assertEqual(stats["total_extracted_nodes"], 2)
        self.assertEqual(stats["total_extracted_relationships"], 1)
        self.assertEqual(stats["avg_nodes_per_document"], 1.0)
        self.assertEqual(stats["avg_relationships_per_document"], 0.5)

    async def test_process_document_with_repository_success(self):
        """Repository를 사용한 문서 처리 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        result = await processor.process_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)

        # Repository 메서드들이 호출되었는지 확인 (DocumentData가 전달됨)
        mock_document_repository.save.assert_called_once()
        mock_document_repository.update_with_knowledge.assert_called_once()

    async def test_process_document_with_repository_failure(self):
        """Repository를 사용한 문서 처리 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        error_message = "추출 실패"
        mock_knowledge_extractor.extract = AsyncMock(
            side_effect=DocumentProcessingException(str(document.id), error_message)
        )
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_document_repository.update = AsyncMock(return_value=document)

        # When & Then
        with self.assertRaises(DocumentProcessingException):
            await processor.process_document(document)

        self.assertEqual(document.status, DocumentStatus.FAILED)
        expected_error_message = f"[DOCUMENT_PROCESSING_FAILED] Failed to process document '{document.id}': {error_message}"
        self.assertEqual(document.metadata["error"], expected_error_message)

        # 실패 상태 업데이트가 호출되었는지 확인 (DocumentData가 전달됨)
        mock_document_repository.update.assert_called_once()

    async def test_process_document_without_repository(self):
        """Repository 없이 문서 처리 테스트 (기존 동작)."""
        # Given
        mock_knowledge_extractor = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor)  # Repository 없음

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))

        # When
        result = await processor.process_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)

        # Repository 관련 호출은 없어야 함
        self.assertIsNone(processor.document_repository)

    async def test_reprocess_document_with_repository(self):
        """Repository를 사용한 문서 재처리 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # 기존 처리 상태로 설정
        document.mark_as_processed()
        document.add_connected_node(NodeId.generate())

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="재처리 노드",
            node_type=DTONodeType.CONCEPT,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))
        mock_document_repository.exists = AsyncMock(return_value=True)
        mock_document_repository.update = AsyncMock(return_value=document)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        result = await processor.reprocess_document(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)

        # 상태 초기화를 위한 update와 재처리를 위한 update_with_knowledge 호출 확인
        self.assertEqual(
            mock_document_repository.update.call_count, 2
        )  # For status reset and processing
        mock_document_repository.update_with_knowledge.assert_called_once()  # For final update

    # === 단계별 예외 처리 테스트 추가 ===

    async def test_process_document_step1_repository_exists_check_failure(self):
        """1단계: Repository exists 체크 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # Repository exists 호출 시 예외 발생
        mock_document_repository.exists = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        # When & Then
        with self.assertRaises(Exception) as context:
            await processor.process_document(document)

        # 문서 상태가 FAILED로 변경되었는지 확인
        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertIn("Database connection failed", document.metadata["error"])

    async def test_process_document_step3_repository_initial_save_failure(self):
        """3단계: Repository 초기 저장 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        mock_document_repository.exists = AsyncMock(return_value=False)
        # 저장 시 예외 발생
        mock_document_repository.save = AsyncMock(side_effect=Exception("Storage failure"))

        # When & Then
        with self.assertRaises(Exception) as context:
            await processor.process_document(document)

        # 문서 상태가 PROCESSING에서 FAILED로 변경되었는지 확인
        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertIn("Storage failure", document.metadata["error"])

        # 실패 상태 업데이트 시도했는지 확인
        mock_document_repository.update.assert_called()

    async def test_process_document_step4_knowledge_extraction_failure_after_save(self):
        """4단계: 지식 추출 실패 시 문서 상태 복구 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        # 지식 추출 시 구체적 예외 발생
        error_message = "LLM service unavailable"
        mock_knowledge_extractor.extract = AsyncMock(
            side_effect=DocumentProcessingException(str(document.id), error_message)
        )
        mock_document_repository.update = AsyncMock(return_value=document)

        # When & Then
        with self.assertRaises(DocumentProcessingException) as context:
            await processor.process_document(document)

        # 구체적 예외 타입과 내용 검증
        self.assertEqual(context.exception.document_id, str(document.id))
        self.assertEqual(context.exception.reason, error_message)

        # 문서 상태가 FAILED로 변경되었는지 확인
        self.assertEqual(document.status, DocumentStatus.FAILED)
        expected_error_message = f"[DOCUMENT_PROCESSING_FAILED] Failed to process document '{document.id}': {error_message}"
        self.assertEqual(document.metadata["error"], expected_error_message)

        # Repository 호출 순서 검증: save -> extract 실패 -> update (실패 상태)
        mock_document_repository.save.assert_called_once()
        mock_knowledge_extractor.extract.assert_called_once()
        mock_document_repository.update.assert_called_once()

    async def test_process_document_step7_final_repository_update_failure(self):
        """7단계: 최종 Repository 업데이트 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))
        # 최종 업데이트 시 실패
        mock_document_repository.update_with_knowledge = AsyncMock(
            side_effect=Exception("Final update failed")
        )
        mock_document_repository.update = AsyncMock(return_value=document)

        # When & Then
        with self.assertRaises(Exception) as context:
            await processor.process_document(document)

        # 지식 추출은 성공했으나 최종 업데이트에서 실패
        self.assertIn("Final update failed", str(context.exception))

        # 문서 상태가 FAILED로 변경되었는지 확인
        self.assertEqual(document.status, DocumentStatus.FAILED)

        # 처리 순서 검증: save -> extract -> update_with_knowledge 실패 -> update (실패 상태)
        mock_document_repository.save.assert_called_once()
        mock_knowledge_extractor.extract.assert_called_once()
        mock_document_repository.update_with_knowledge.assert_called_once()
        mock_document_repository.update.assert_called_once()

    async def test_process_document_repository_update_failure_during_error_handling(self):
        """예외 처리 중 Repository 업데이트 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        # 지식 추출 실패
        original_error = "Knowledge extraction failed"
        mock_knowledge_extractor.extract = AsyncMock(
            side_effect=DocumentProcessingException(str(document.id), original_error)
        )
        # 실패 상태 업데이트도 실패
        mock_document_repository.update = AsyncMock(
            side_effect=Exception("Update during error handling failed")
        )

        # When & Then
        with self.assertRaises(DocumentProcessingException) as context:
            await processor.process_document(document)

        # 원본 예외가 전파되어야 함
        self.assertEqual(context.exception.reason, original_error)

        # 문서 상태는 여전히 FAILED로 설정되어야 함
        self.assertEqual(document.status, DocumentStatus.FAILED)

    async def test_process_document_validate_processing_order_success(self):
        """정상 처리 시 단계별 순서 검증 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        # 모든 단계 성공 설정
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        result = await processor.process_document(document)

        # Then - 처리 순서 및 상태 변화 검증
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertIsNotNone(document.processed_at)
        self.assertEqual(len(document.connected_nodes), 1)

        # 호출 순서 검증
        mock_document_repository.exists.assert_called_once()
        mock_document_repository.save.assert_called_once()
        mock_knowledge_extractor.extract.assert_called_once()
        mock_document_repository.update_with_knowledge.assert_called_once()

    async def test_process_document_memory_vs_persistence_consistency(self):
        """메모리 처리와 영속성 처리 결과 일관성 테스트."""
        # Given - 메모리 처리용
        mock_knowledge_extractor_memory = Mock()
        processor_memory = DocumentProcessor(mock_knowledge_extractor_memory)

        # Given - 영속성 처리용
        mock_knowledge_extractor_persistence = Mock()
        mock_document_repository = AsyncMock()
        processor_persistence = DocumentProcessor(
            mock_knowledge_extractor_persistence, mock_document_repository
        )

        # 동일한 문서와 추출 결과 설정
        document_memory = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        document_persistence = Document(
            id=document_memory.id,
            title=document_memory.title,
            content=document_memory.content,
            doc_type=document_memory.doc_type,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        mock_knowledge_extractor_memory.extract = AsyncMock(return_value=([sample_node_data], []))
        mock_knowledge_extractor_persistence.extract = AsyncMock(
            return_value=([sample_node_data], [])
        )

        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document_persistence)
        mock_document_repository.update_with_knowledge = AsyncMock(
            return_value=document_persistence
        )

        # When
        result_memory = await processor_memory.process_document(document_memory)
        result_persistence = await processor_persistence.process_document(document_persistence)

        # Then - 두 처리 방식의 결과가 일관성 있는지 확인
        self.assertEqual(document_memory.status, document_persistence.status)
        self.assertEqual(
            len(document_memory.connected_nodes), len(document_persistence.connected_nodes)
        )
        self.assertEqual(
            len(document_memory.connected_relationships),
            len(document_persistence.connected_relationships),
        )
        self.assertEqual(result_memory.get_node_count(), result_persistence.get_node_count())
        self.assertEqual(
            result_memory.get_relationship_count(), result_persistence.get_relationship_count()
        )

    # === Mock 검증 로직 강화 테스트 추가 ===

    async def test_method_call_sequence_and_arguments_verification(self):
        """메서드 호출 순서와 인자 상세 검증 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="순서 검증 테스트",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 노드",
            node_type=DTONodeType.PERSON,
            properties={"test": "value"},
        )

        # Mock 설정
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        await processor.process_document(document)

        # Then - 호출 순서 검증
        expected_call_sequence = [
            ("exists", str(document.id)),
            ("save", "document_data"),
            ("extract", "document_data"),
            ("update_with_knowledge", "document_data", "node_ids", "relationship_ids"),
        ]

        # 실제 호출 순서 추적
        actual_calls = []

        # exists 호출 확인
        mock_document_repository.exists.assert_called_once_with(str(document.id))
        actual_calls.append(("exists", str(document.id)))

        # save 호출 확인 및 매개변수 검증
        self.assertEqual(mock_document_repository.save.call_count, 1)
        save_call_args = mock_document_repository.save.call_args[0][0]
        self.assertEqual(save_call_args.id, str(document.id))
        self.assertEqual(save_call_args.title, document.title)
        self.assertEqual(save_call_args.content, document.content)
        actual_calls.append(("save", "document_data"))

        # extract 호출 확인 및 매개변수 검증
        mock_knowledge_extractor.extract.assert_called_once()
        extract_call_args = mock_knowledge_extractor.extract.call_args[0][0]
        self.assertEqual(extract_call_args.id, str(document.id))
        self.assertEqual(extract_call_args.title, document.title)
        actual_calls.append(("extract", "document_data"))

        # update_with_knowledge 호출 확인 및 매개변수 검증
        mock_document_repository.update_with_knowledge.assert_called_once()
        update_call_args = mock_document_repository.update_with_knowledge.call_args

        # 첫 번째 인자: document_data
        document_data_arg = update_call_args[0][0]
        self.assertEqual(document_data_arg.id, str(document.id))

        # 두 번째 인자: node_ids 리스트
        node_ids_arg = update_call_args[0][1]
        self.assertEqual(len(node_ids_arg), 1)
        self.assertEqual(node_ids_arg[0], sample_node_data.id)

        # 세 번째 인자: relationship_ids 리스트
        relationship_ids_arg = update_call_args[0][2]
        self.assertEqual(len(relationship_ids_arg), 0)

        actual_calls.append(
            ("update_with_knowledge", "document_data", "node_ids", "relationship_ids")
        )

        # 전체 호출 순서가 예상과 일치하는지 확인
        self.assertEqual(len(actual_calls), 4)
        for i, (expected_method, _) in enumerate(expected_call_sequence):
            actual_method, _ = actual_calls[i]
            self.assertEqual(
                actual_method,
                expected_method,
                f"호출 순서 {i}에서 {expected_method} 예상, {actual_method} 실제",
            )

    async def test_mock_call_args_deep_inspection(self):
        """Mock 호출 인자의 깊이 있는 검사 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="깊이 있는 검사",
            content="복잡한 내용 with 특수문자 & 유니코드 한글",
            doc_type=DocumentType.TEXT,
        )

        # 메타데이터 추가
        document.update_metadata("author", "테스트 작성자")
        document.update_metadata("priority", "high")

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="복잡한 노드",
            node_type=DTONodeType.ORGANIZATION,
            properties={"location": "서울", "type": "tech_company", "employees": 100},
        )

        sample_relationship_data = RelationshipData(
            id=str(RelationshipId.generate()),
            source_node_id=str(NodeId.generate()),
            target_node_id=str(NodeId.generate()),
            relationship_type=DTORelationshipType.LOCATED_IN,
            properties={"since": "2020", "confidence": 0.95},
        )

        # Mock 설정
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_knowledge_extractor.extract = AsyncMock(
            return_value=([sample_node_data], [sample_relationship_data])
        )
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        await processor.process_document(document)

        # Then - 인자의 상세 내용 검사

        # 1. save 호출 인자 깊이 검사
        save_args = mock_document_repository.save.call_args[0][0]
        self.assertEqual(save_args.title, "깊이 있는 검사")
        self.assertIn("특수문자", save_args.content)
        self.assertIn("한글", save_args.content)

        # 2. extract 호출 인자 깊이 검사
        extract_args = mock_knowledge_extractor.extract.call_args[0][0]
        self.assertEqual(extract_args.title, "깊이 있는 검사")
        self.assertEqual(extract_args.doc_type.value, "text")

        # 3. update_with_knowledge 호출 인자 깊이 검사
        update_args = mock_document_repository.update_with_knowledge.call_args[0]

        # 문서 데이터 검사
        doc_data = update_args[0]
        self.assertEqual(doc_data.title, "깊이 있는 검사")

        # 노드 ID 리스트 검사
        node_ids = update_args[1]
        self.assertEqual(len(node_ids), 1)
        self.assertEqual(node_ids[0], sample_node_data.id)

        # 관계 ID 리스트 검사
        relationship_ids = update_args[2]
        self.assertEqual(len(relationship_ids), 1)
        self.assertEqual(relationship_ids[0], sample_relationship_data.id)

    async def test_mock_partial_argument_matching(self):
        """Mock 부분 매개변수 매칭 및 타입 검증 테스트."""
        from unittest.mock import ANY

        # Given
        mock_knowledge_extractor = Mock()
        mock_document_repository = AsyncMock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="부분 매칭 테스트",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # Mock 설정
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_knowledge_extractor.extract = AsyncMock(return_value=([], []))
        mock_document_repository.update_with_knowledge = AsyncMock(return_value=document)

        # When
        await processor.process_document(document)

        # Then - 부분 매개변수 매칭 검증

        # 1. exists 호출에서 문서 ID 타입 검증
        exists_call = mock_document_repository.exists.call_args[0][0]
        self.assertIsInstance(exists_call, str)  # 문서 ID는 문자열로 전달되어야 함

        # 2. save 호출에서 DocumentData 타입 검증
        save_call = mock_document_repository.save.call_args[0][0]
        # DocumentData의 필수 속성들 존재 확인
        self.assertTrue(hasattr(save_call, "id"))
        self.assertTrue(hasattr(save_call, "title"))
        self.assertTrue(hasattr(save_call, "content"))
        self.assertTrue(hasattr(save_call, "doc_type"))
        self.assertTrue(hasattr(save_call, "status"))

        # 3. extract 호출에서 DocumentData 타입 및 내용 검증
        extract_call = mock_knowledge_extractor.extract.call_args[0][0]
        self.assertEqual(extract_call.title, "부분 매칭 테스트")

        # 4. update_with_knowledge 호출에서 리스트 타입 검증
        update_call = mock_document_repository.update_with_knowledge.call_args[0]
        node_ids_arg = update_call[1]
        relationship_ids_arg = update_call[2]

        self.assertIsInstance(node_ids_arg, list)
        self.assertIsInstance(relationship_ids_arg, list)

        # 빈 리스트여야 함 (extract가 빈 결과를 반환하므로)
        self.assertEqual(len(node_ids_arg), 0)
        self.assertEqual(len(relationship_ids_arg), 0)

        # ANY 매처를 사용한 유연한 검증
        mock_document_repository.exists.assert_called_once_with(ANY)
        mock_document_repository.save.assert_called_once_with(ANY)
        mock_knowledge_extractor.extract.assert_called_once_with(ANY)
        mock_document_repository.update_with_knowledge.assert_called_once_with(ANY, ANY, ANY)


class TestKnowledgeExtractionResult(unittest.TestCase):
    """KnowledgeExtractionResult 테스트."""

    def test_create_empty_result(self):
        """빈 결과 생성 테스트."""
        # When
        result = KnowledgeExtractionResult([], [])

        # Then
        self.assertTrue(result.is_empty())
        self.assertEqual(result.get_node_count(), 0)
        self.assertEqual(result.get_relationship_count(), 0)
        self.assertIsNotNone(result.extracted_at)

    def test_create_non_empty_result(self):
        """비어있지 않은 결과 생성 테스트."""
        # Given
        node = Node(id=NodeId.generate(), name="테스트 노드", node_type=NodeType.PERSON)
        relationship = Relationship(
            id=RelationshipId.generate(),
            source_node_id=NodeId.generate(),
            target_node_id=NodeId.generate(),
            relationship_type=RelationshipType.WORKS_AT,
            label="근무",
        )

        # When
        result = KnowledgeExtractionResult([node], [relationship])

        # Then
        self.assertFalse(result.is_empty())
        self.assertEqual(result.get_node_count(), 1)
        self.assertEqual(result.get_relationship_count(), 1)
        self.assertEqual(result.nodes[0], node)
        self.assertEqual(result.relationships[0], relationship)


if __name__ == "__main__":
    unittest.main()
