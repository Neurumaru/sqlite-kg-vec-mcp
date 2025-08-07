"""
DocumentProcessor.process 메서드 단위 테스트.
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


class TestDocumentProcessorProcess(unittest.IsolatedAsyncioTestCase):
    """DocumentProcessor.process 메서드 테스트."""

    async def test_success(self):
        """문서 처리 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper,
        )

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

        sample_rel_data = RelationshipData(
            id=str(RelationshipId.generate()),
            source_node_id=str(NodeId.generate()),
            target_node_id=str(NodeId.generate()),
            relationship_type=DTORelationshipType.CONTAINS,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(
            return_value=([sample_node_data], [sample_rel_data])
        )

        # Create actual entity instances that the mappers should return
        sample_node = Node(
            id=NodeId(sample_node_data.id),
            name=sample_node_data.name,
            node_type=NodeType.PERSON,
            properties=sample_node_data.properties,
        )

        sample_relationship = Relationship(
            id=RelationshipId(sample_rel_data.id),
            source_node_id=NodeId(sample_rel_data.source_node_id),
            target_node_id=NodeId(sample_rel_data.target_node_id),
            relationship_type=RelationshipType.CONTAINS,
            label="CONTAINS",
            properties=sample_rel_data.properties,
        )

        # Mock the mappers to return actual entity instances
        mock_node_mapper.from_data.return_value = sample_node
        mock_relationship_mapper.from_data.return_value = sample_relationship

        # When
        result = await processor.process(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertIsNotNone(document.processed_at)
        # document_data로 변환되어 호출되므로 assert_called_once()만 확인
        mock_knowledge_extractor.extract.assert_called_once()

    async def test_document_processing_exception_when_extraction_fails(self):
        """문서 처리 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper,
        )

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
            await processor.process(document)

        self.assertEqual(document.status, DocumentStatus.FAILED)
        expected_error_message = f"[DOCUMENT_PROCESSING_FAILED] Failed to process document '{document.id}': {error_message}"
        self.assertEqual(document.metadata["error"], expected_error_message)

    async def test_success_when_empty_extraction_result(self):
        """빈 추출 결과로 문서 처리 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper,
        )

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # 빈 추출 결과
        mock_knowledge_extractor.extract = AsyncMock(return_value=([], []))

        # When
        result = await processor.process(document)

        # Then
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.relationships), 0)
        self.assertTrue(result.is_empty())
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertIsNotNone(document.processed_at)


if __name__ == "__main__":
    unittest.main()
