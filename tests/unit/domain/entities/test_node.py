"""
Node 엔티티 단위 테스트.
"""

import unittest
from datetime import datetime

from src.domain.entities.node import Node, NodeType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId


class TestNode(unittest.TestCase):
    """Node 엔티티 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.node_id = NodeId.generate()
        self.name = "테스트 노드"
        self.node_type = NodeType.PERSON
        self.description = "테스트용 노드입니다."

    def test_create_node(self):
        """노드 생성 테스트."""
        node = Node(
            id=self.node_id,
            name=self.name,
            node_type=self.node_type,
            description=self.description,
        )

        self.assertEqual(node.id, self.node_id)
        self.assertEqual(node.name, self.name)
        self.assertEqual(node.node_type, self.node_type)
        self.assertEqual(node.description, self.description)
        self.assertIsInstance(node.created_at, datetime)

    def test_node_types(self):
        """노드 타입 테스트."""
        person_node = Node(id=NodeId.generate(), name="홍길동", node_type=NodeType.PERSON)

        organization_node = Node(
            id=NodeId.generate(), name="삼성전자", node_type=NodeType.ORGANIZATION
        )

        self.assertEqual(person_node.node_type, NodeType.PERSON)
        self.assertEqual(organization_node.node_type, NodeType.ORGANIZATION)

    def test_update_property(self):
        """노드 속성 추가 테스트."""
        node = Node(id=self.node_id, name=self.name, node_type=self.node_type)

        node.update_property("age", 30)

        self.assertEqual(node.properties["age"], 30)

    def test_remove_property(self):
        """노드 속성 제거 테스트."""
        node = Node(id=self.node_id, name=self.name, node_type=self.node_type)
        node.update_property("age", 30)

        node.remove_property("age")

        self.assertNotIn("age", node.properties)

    def test_add_source_document(self):
        """출처 문서 추가 테스트."""
        node = Node(id=self.node_id, name=self.name, node_type=self.node_type)
        doc_id = DocumentId.generate()
        context = "문서에서 언급됨"

        node.add_source_document(doc_id, context)

        self.assertIn(doc_id, node.source_documents)
        # context는 extraction_metadata에 저장됨
        context_key = f"context_{doc_id}"
        self.assertIn(context_key, node.extraction_metadata)


if __name__ == "__main__":
    unittest.main()
