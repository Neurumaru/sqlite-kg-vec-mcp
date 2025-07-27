"""
Document 엔티티 단위 테스트.
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId


class TestDocument(unittest.TestCase):
    """Document 엔티티 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.document_id = DocumentId.generate()
        self.title = "테스트 문서"
        self.content = "이것은 테스트 문서 내용입니다."
        
    def test_create_document(self):
        """문서 생성 테스트."""
        document = Document(
            id=self.document_id,
            title=self.title,
            content=self.content,
            doc_type=DocumentType.TEXT
        )
        
        self.assertEqual(document.id, self.document_id)
        self.assertEqual(document.title, self.title)
        self.assertEqual(document.content, self.content)
        self.assertEqual(document.status, DocumentStatus.PENDING)
        self.assertIsInstance(document.created_at, datetime)
        
    def test_mark_as_processing(self):
        """문서 처리 중 상태 변경 테스트."""
        document = Document(id=self.document_id, title=self.title, content=self.content, doc_type=DocumentType.TEXT)
        
        document.mark_as_processing()
        
        self.assertEqual(document.status, DocumentStatus.PROCESSING)
        
    def test_mark_as_processed(self):
        """문서 처리 완료 상태 변경 테스트."""
        document = Document(id=self.document_id, title=self.title, content=self.content, doc_type=DocumentType.TEXT)
        
        document.mark_as_processed()
        
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertIsNotNone(document.processed_at)
        
    def test_mark_as_failed(self):
        """문서 처리 실패 상태 변경 테스트."""
        document = Document(id=self.document_id, title=self.title, content=self.content, doc_type=DocumentType.TEXT)
        error_message = "처리 실패"
        
        document.mark_as_failed(error_message)
        
        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertEqual(document.metadata["error"], error_message)
        
    def test_add_connected_node(self):
        """연결된 노드 추가 테스트."""
        document = Document(id=self.document_id, title=self.title, content=self.content, doc_type=DocumentType.TEXT)
        from src.domain.value_objects.node_id import NodeId
        node_id = NodeId.generate()
        
        document.add_connected_node(node_id)
        
        self.assertIn(node_id, document.connected_nodes)
        
    def test_remove_connected_node(self):
        """연결된 노드 제거 테스트."""
        document = Document(id=self.document_id, title=self.title, content=self.content, doc_type=DocumentType.TEXT)
        from src.domain.value_objects.node_id import NodeId
        node_id = NodeId.generate()
        
        document.add_connected_node(node_id)
        document.remove_connected_node(node_id)
        
        self.assertNotIn(node_id, document.connected_nodes)


if __name__ == "__main__":
    unittest.main()