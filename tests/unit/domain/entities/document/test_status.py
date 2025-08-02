"""
Document 엔티티 상태 관리 테스트.
"""

import time
import unittest
from datetime import datetime

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId


class TestDocumentStatus(unittest.TestCase):
    """Document 엔티티 상태 관리 테스트."""

    def test_success_when_mark_as_processing(self):
        """문서 처리 중 상태 변경 성공 테스트."""

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        original_updated_at = document.updated_at

        # 시간 차이를 보장하기 위한 작은 딜레이
        time.sleep(0.001)

        # When
        document.mark_as_processing()

        # Then
        self.assertEqual(document.status, DocumentStatus.PROCESSING)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertIsNone(document.processed_at)

    def test_success_when_mark_as_processed(self):
        """문서 처리 완료 상태 변경 성공 테스트."""

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        original_updated_at = document.updated_at

        # 시간 차이를 보장하기 위한 작은 딜레이
        time.sleep(0.001)

        # When
        document.mark_as_processed()

        # Then
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertIsInstance(document.processed_at, datetime)
        self.assertTrue(document.is_processed())

    def test_success_when_mark_as_failed(self):
        """문서 처리 실패 상태 변경 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        error_message = "처리 중 오류 발생"
        original_updated_at = document.updated_at

        # When
        document.mark_as_failed(error_message)

        # Then
        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertEqual(document.metadata["error"], error_message)
        self.assertFalse(document.is_processed())


if __name__ == "__main__":
    unittest.main()
