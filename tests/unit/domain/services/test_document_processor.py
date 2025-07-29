"""
DocumentProcessor 도메인 서비스 단위 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock

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
from src.dto.node import NodeData
from src.dto.node import NodeType as DTONodeType
from src.dto.relationship import RelationshipData
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
        mock_knowledge_extractor.extract = AsyncMock(side_effect=Exception(error_message))

        # When & Then
        with self.assertRaises(Exception):
            await processor.process_document(document)

        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertEqual(document.metadata["error"], error_message)

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
        mock_document_repository = Mock()
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
        mock_document_repository = Mock()
        processor = DocumentProcessor(mock_knowledge_extractor, mock_document_repository)

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        error_message = "추출 실패"
        mock_knowledge_extractor.extract = AsyncMock(side_effect=Exception(error_message))
        mock_document_repository.exists = AsyncMock(return_value=False)
        mock_document_repository.save = AsyncMock(return_value=document)
        mock_document_repository.update = AsyncMock(return_value=document)

        # When & Then
        with self.assertRaises(Exception):
            await processor.process_document(document)

        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertEqual(document.metadata["error"], error_message)

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
        mock_document_repository = Mock()
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
        )  # 상태 초기화 + 재처리 중 상태 업데이트
        mock_document_repository.update_with_knowledge.assert_called_once()  # 재처리 완료


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
