"""
Relationship 엔티티 단위 테스트.
"""

import unittest
from datetime import datetime

from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.node_id import NodeId


class TestRelationship(unittest.TestCase):
    """Relationship 엔티티 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.relationship_id = RelationshipId.generate()
        self.source_node_id = NodeId.generate()
        self.target_node_id = NodeId.generate()
        self.relationship_type = RelationshipType.WORKS_AT
        self.label = "근무"
        
    def test_create_relationship(self):
        """관계 생성 테스트."""
        relationship = Relationship(
            id=self.relationship_id,
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            relationship_type=self.relationship_type,
            label=self.label
        )
        
        self.assertEqual(relationship.id, self.relationship_id)
        self.assertEqual(relationship.source_node_id, self.source_node_id)
        self.assertEqual(relationship.target_node_id, self.target_node_id)
        self.assertEqual(relationship.relationship_type, self.relationship_type)
        self.assertEqual(relationship.label, self.label)
        self.assertIsInstance(relationship.created_at, datetime)
        
    def test_relationship_types(self):
        """관계 타입 테스트."""
        works_for = Relationship(
            id=RelationshipId.generate(),
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            relationship_type=RelationshipType.WORKS_AT,
            label="근무"
        )
        
        creates = Relationship(
            id=RelationshipId.generate(),
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            relationship_type=RelationshipType.CREATES,
            label="생성"
        )
        
        self.assertEqual(works_for.relationship_type, RelationshipType.WORKS_AT)
        self.assertEqual(creates.relationship_type, RelationshipType.CREATES)
        
    def test_set_confidence(self):
        """신뢰도 설정 테스트."""
        relationship = Relationship(
            id=self.relationship_id,
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            relationship_type=self.relationship_type,
            label=self.label
        )
        
        relationship.update_confidence(0.8)
        
        self.assertEqual(relationship.confidence, 0.8)
        
    def test_add_source_document(self):
        """출처 문서 추가 테스트."""
        relationship = Relationship(
            id=self.relationship_id,
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            relationship_type=self.relationship_type,
            label=self.label
        )
        
        from src.domain.value_objects.document_id import DocumentId
        doc_id = DocumentId.generate()
        context = "문서에서 언급됨"
        sentence = "홍길동이 삼성전자에서 근무한다."
        
        relationship.add_source_document(doc_id, context, sentence)
        
        self.assertIn(doc_id, relationship.source_documents)
        # context와 sentence는 하나의 키에 딕셔너리로 저장됨
        context_key = f"context_{doc_id}"
        self.assertIn(context_key, relationship.extraction_metadata)
        self.assertEqual(relationship.extraction_metadata[context_key]["context"], context)
        self.assertEqual(relationship.extraction_metadata[context_key]["sentence"], sentence)


if __name__ == "__main__":
    unittest.main()