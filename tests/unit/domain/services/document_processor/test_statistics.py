"""
DocumentProcessor.get_processing_statistics 메서드 단위 테스트.
"""

import unittest
from unittest.mock import Mock

from src.domain.entities.document import Document, DocumentType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentProcessorGetProcessingStatistics(unittest.TestCase):
    """DocumentProcessor.get_processing_statistics 메서드 테스트."""

    def test_success(self):
        """문서 처리 통계 계산 성공 테스트."""
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

        # 연결된 노드와 관계 추가
        node_id_1 = NodeId.generate()
        node_id_2 = NodeId.generate()
        rel_id_1 = RelationshipId.generate()

        document.connected_nodes = [node_id_1, node_id_2]
        document.connected_relationships = [rel_id_1]

        # When
        stats = processor.get_processing_statistics([document])

        # Then
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["processed"], 0)
        self.assertEqual(stats["processing"], 0)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["total_extracted_nodes"], 2)
        self.assertEqual(stats["total_extracted_relationships"], 1)


if __name__ == "__main__":
    unittest.main()
