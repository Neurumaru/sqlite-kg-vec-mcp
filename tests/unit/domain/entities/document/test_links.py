"""
Document 엔티티 연결 관리 테스트.
"""

import time
import unittest

from src.domain.entities.document import Document, DocumentType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentLinks(unittest.TestCase):
    """Document 엔티티 연결 관리 테스트."""

    def test_success_when_add_connected_node(self):
        """연결된 노드 추가 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        original_updated_at = document.updated_at

        # When
        document.add_connected_node(node_id)

        # Then
        self.assertIn(node_id, document.connected_nodes)
        self.assertEqual(len(document.connected_nodes), 1)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertTrue(document.has_connected_elements())

    def test_success_when_add_connected_node_duplicate_ignored(self):
        """중복된 노드 추가 시 무시되는지 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)
        original_count = len(document.connected_nodes)

        # When
        document.add_connected_node(node_id)  # 중복 추가

        # Then
        self.assertEqual(len(document.connected_nodes), original_count)
        self.assertEqual(document.connected_nodes.count(node_id), 1)

    def test_success_when_add_connected_relationship(self):
        """연결된 관계 추가 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        original_updated_at = document.updated_at

        # When
        document.add_connected_relationship(relationship_id)

        # Then
        self.assertIn(relationship_id, document.connected_relationships)
        self.assertEqual(len(document.connected_relationships), 1)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertTrue(document.has_connected_elements())

    def test_success_when_remove_connected_node(self):
        """연결된 노드 제거 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)
        original_updated_at = document.updated_at

        # When
        time.sleep(0.001)  # 타이밍 차이 보장
        document.remove_connected_node(node_id)

        # Then
        self.assertNotIn(node_id, document.connected_nodes)
        self.assertEqual(len(document.connected_nodes), 0)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_success_when_remove_connected_node_not_exists(self):
        """존재하지 않는 노드 제거 시 무시되는지 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        original_updated_at = document.updated_at
        original_count = len(document.connected_nodes)

        # When
        document.remove_connected_node(node_id)

        # Then
        self.assertEqual(len(document.connected_nodes), original_count)
        self.assertEqual(document.updated_at, original_updated_at)

    def test_success_when_remove_connected_relationship(self):
        """연결된 관계 제거 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        document.add_connected_relationship(relationship_id)
        original_updated_at = document.updated_at

        # When
        document.remove_connected_relationship(relationship_id)

        # Then
        self.assertNotIn(relationship_id, document.connected_relationships)
        self.assertEqual(len(document.connected_relationships), 0)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_true_when_has_connected_elements_with_nodes(self):
        """노드가 연결된 경우 연결 요소 존재 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)

        # When & Then
        self.assertTrue(document.has_connected_elements())

    def test_true_when_has_connected_elements_with_relationships(self):
        """관계가 연결된 경우 연결 요소 존재 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        document.add_connected_relationship(relationship_id)

        # When & Then
        self.assertTrue(document.has_connected_elements())

    def test_false_when_has_connected_elements_empty(self):
        """연결 요소가 없는 경우 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When & Then
        self.assertFalse(document.has_connected_elements())


if __name__ == "__main__":
    unittest.main()
