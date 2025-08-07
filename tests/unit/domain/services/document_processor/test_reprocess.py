"""
DocumentProcessor.reprocess_document 메서드 단위 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.dto.node import NodeData
from src.dto.node import NodeType as DTONodeType


class TestDocumentProcessorReprocessDocument(unittest.IsolatedAsyncioTestCase):
    """DocumentProcessor.reprocess_document 메서드 테스트."""

    async def test_success(self):
        """문서 재처리 성공 테스트."""
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
        document.mark_as_processed()

        sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="재처리 노드",
            node_type=DTONodeType.PERSON,
            properties={},
        )

        mock_knowledge_extractor.extract = AsyncMock(return_value=([sample_node_data], []))

        # Create actual entity instance that the mapper should return
        sample_node = Node(
            id=NodeId(sample_node_data.id),
            name=sample_node_data.name,
            node_type=NodeType.PERSON,
            properties=sample_node_data.properties,
        )

        # Mock the mapper to return actual entity instance
        mock_node_mapper.from_data.return_value = sample_node

        # When
        result = await processor.reprocess_document(document)

        # Then
        self.assertIsNotNone(result)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertEqual(len(document.connected_nodes), 1)


if __name__ == "__main__":
    unittest.main()
