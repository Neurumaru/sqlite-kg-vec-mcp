"""
KnowledgeExtractionResult 클래스 단위 테스트.
"""

import unittest

from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.document_processor import KnowledgeExtractionResult
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestKnowledgeExtractionResult(unittest.TestCase):
    """KnowledgeExtractionResult 클래스 테스트."""

    def test_create_empty_result(self):
        """빈 추출 결과 생성 테스트."""
        # When
        result = KnowledgeExtractionResult([], [])

        # Then
        self.assertTrue(result.is_empty())
        self.assertEqual(result.get_node_count(), 0)
        self.assertEqual(result.get_relationship_count(), 0)

    def test_create_non_empty_result(self):
        """비어있지 않은 추출 결과 생성 테스트."""
        # Given
        node = Node(
            id=NodeId.generate(),
            name="테스트 노드",
            node_type=NodeType.PERSON,
        )

        relationship = Relationship(
            id=RelationshipId.generate(),
            source_node_id=NodeId.generate(),
            target_node_id=NodeId.generate(),
            relationship_type=RelationshipType.SIMILAR_TO,
            label="테스트 관계",
        )

        # When
        result = KnowledgeExtractionResult([node], [relationship])

        # Then
        self.assertFalse(result.is_empty())
        self.assertEqual(result.get_node_count(), 1)
        self.assertEqual(result.get_relationship_count(), 1)
        self.assertEqual(result.nodes[0], node)
        self.assertEqual(result.relationships[0], relationship)


if __name__ == "__main__":
    unittest.main()
