"""
Document 엔티티 메타데이터 및 유틸리티 테스트.
"""

import time
import unittest

from src.domain.entities.document import Document, DocumentType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentMetadata(unittest.TestCase):
    """Document 엔티티 메타데이터 및 유틸리티 테스트."""

    def test_success_when_get_word_count(self):
        """문서 단어 수 계산 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트",
            content="이것은 테스트 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        # When
        word_count = document.get_word_count()

        # Then
        self.assertEqual(word_count, 3)  # "이것은", "테스트", "문서입니다."

    def test_success_when_get_char_count(self):
        """문서 문자 수 계산 성공 테스트."""
        content = "테스트 문서"
        document = Document(
            id=DocumentId.generate(), title="테스트", content=content, doc_type=DocumentType.TEXT
        )

        # When
        char_count = document.get_char_count()

        # Then
        self.assertEqual(char_count, len(content))

    def test_success_when_update_metadata(self):
        """메타데이터 업데이트 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        key = "author"
        value = "홍길동"
        original_updated_at = document.updated_at

        # When
        document.update_metadata(key, value)

        # Then
        self.assertEqual(document.metadata[key], value)
        self.assertGreater(document.updated_at, original_updated_at)


class TestDocumentComplexOperations(unittest.TestCase):
    """Document 엔티티 복합 연산 테스트."""

    def test_success_when_complex_timestamp_operations(self):
        """복잡한 타임스탬프 연산 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="이것은 복잡한 테스트 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        initial_created_at = document.created_at
        initial_updated_at = document.updated_at

        # created_at should equal updated_at initially
        self.assertEqual(initial_created_at, initial_updated_at)

        def mark_as_processing_op():
            document.mark_as_processing()

        def add_node_op():
            document.add_connected_node(NodeId.generate())

        def add_relationship_op():
            document.add_connected_relationship(RelationshipId.generate())

        def update_metadata_op():
            document.update_metadata("test", "value")

        def mark_as_processed_op():
            document.mark_as_processed()

        operations = [
            ("mark_as_processing", mark_as_processing_op),
            ("add_node", add_node_op),
            ("add_relationship", add_relationship_op),
            ("update_metadata", update_metadata_op),
            ("mark_as_processed", mark_as_processed_op),
        ]

        previous_updated_at = initial_updated_at

        for operation_name, operation in operations:
            time.sleep(0.001)  # Ensure timestamp difference
            operation()

            # Check that updated_at increased
            self.assertGreater(
                document.updated_at,
                previous_updated_at,
                f"updated_at should increase after {operation_name}",
            )

            # Check that created_at remains unchanged
            self.assertEqual(
                document.created_at,
                initial_created_at,
                f"created_at should not change after {operation_name}",
            )

            previous_updated_at = document.updated_at

        # Final verifications
        self.assertTrue(document.is_processed())
        self.assertTrue(document.has_connected_elements())
        self.assertIsNotNone(document.processed_at)
        self.assertIn("test", document.metadata)


if __name__ == "__main__":
    unittest.main()
