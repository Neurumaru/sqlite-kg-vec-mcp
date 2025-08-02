"""
Document 엔티티 생성 및 검증 테스트.
"""

import unittest
from datetime import datetime

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId


class TestDocumentCreation(unittest.TestCase):
    """Document 엔티티 생성 및 검증 테스트."""

    def test_success(self):
        """문서 생성 성공 테스트."""
        # When
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="이것은 테스트 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        # Then
        self.assertEqual(document.title, "테스트 문서")
        self.assertEqual(document.content, "이것은 테스트 문서입니다.")
        self.assertEqual(document.doc_type, DocumentType.TEXT)
        self.assertEqual(document.status, DocumentStatus.PENDING)
        self.assertIsInstance(document.created_at, datetime)
        self.assertIsInstance(document.updated_at, datetime)
        self.assertIsNone(document.processed_at)
        self.assertEqual(len(document.connected_nodes), 0)
        self.assertEqual(len(document.connected_relationships), 0)
        self.assertEqual(len(document.metadata), 0)

    def test_value_error_when_empty_title(self):
        """빈 제목으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="",
                content="테스트 내용",
                doc_type=DocumentType.TEXT,
            )

    def test_value_error_when_whitespace_title(self):
        """공백 제목으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="   ",
                content="테스트 내용",
                doc_type=DocumentType.TEXT,
            )

    def test_value_error_when_empty_content(self):
        """빈 내용으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="테스트 제목",
                content="",
                doc_type=DocumentType.TEXT,
            )

    def test_value_error_when_whitespace_content(self):
        """공백 내용으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="테스트 제목",
                content="   ",
                doc_type=DocumentType.TEXT,
            )


if __name__ == "__main__":
    unittest.main()
