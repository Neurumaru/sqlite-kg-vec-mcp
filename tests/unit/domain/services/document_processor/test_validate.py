"""
DocumentProcessor.validate_document_for_processing 메서드 단위 테스트.
"""

import unittest
from unittest.mock import Mock

from src.domain.entities.document import Document, DocumentType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId


class TestDocumentProcessorValidateDocumentForProcessing(unittest.TestCase):
    """DocumentProcessor.validate_document_for_processing 메서드 테스트."""

    def test_success(self):
        """문서 처리 검증 성공 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper
        )

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertTrue(result)

    def test_false_when_document_already_processing(self):
        """이미 처리 중인 문서 검증 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper
        )

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        document.mark_as_processing()

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertFalse(result)

    def test_false_when_document_already_processed(self):
        """이미 처리 완료된 문서 검증 실패 테스트."""
        # Given
        mock_knowledge_extractor = Mock()
        mock_document_mapper = Mock()
        mock_node_mapper = Mock()
        mock_relationship_mapper = Mock()
        processor = DocumentProcessor(
            mock_knowledge_extractor,
            mock_document_mapper,
            mock_node_mapper,
            mock_relationship_mapper
        )

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        document.mark_as_processed()

        # When
        result = processor.validate_document_for_processing(document)

        # Then
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
