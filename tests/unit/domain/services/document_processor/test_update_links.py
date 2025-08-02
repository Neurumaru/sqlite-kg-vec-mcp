"""
DocumentProcessor.update_document_links 메서드 단위 테스트.
"""

import unittest
from unittest.mock import Mock

from src.domain.entities.document import Document, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentProcessorUpdateDocumentLinks(unittest.TestCase):
    """DocumentProcessor.update_document_links 메서드 테스트."""

    def test_success(self):
        """문서 링크 업데이트 성공 테스트."""
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

        # 기존 연결된 노드와 관계
        node_id_1 = NodeId.generate()
        rel_id_1 = RelationshipId.generate()
        document.connected_nodes = [node_id_1]
        document.connected_relationships = [rel_id_1]

        # 새로운 노드와 관계
        node_id_2 = NodeId.generate()
        rel_id_2 = RelationshipId.generate()

        _ = Node(
            id=node_id_2,
            name="새 노드",
            node_type=NodeType.PERSON,
        )

        _ = Relationship(
            id=rel_id_2,
            source_node_id=node_id_1,
            target_node_id=node_id_2,
            relationship_type=RelationshipType.SIMILAR_TO,
            label="새 관계",
        )

        # When
        processor.update_document_links(
            document,
            added_nodes=[node_id_2],
            removed_relationships=[rel_id_1],
            added_relationships=[rel_id_2],
        )

        # Then
        # 기존 노드는 유지되고 새 노드가 추가되어야 함
        self.assertIn(node_id_1, document.connected_nodes)
        self.assertIn(node_id_2, document.connected_nodes)
        # 기존 관계는 제거되고 새 관계가 추가되어야 함
        self.assertNotIn(rel_id_1, document.connected_relationships)
        self.assertIn(rel_id_2, document.connected_relationships)


if __name__ == "__main__":
    unittest.main()
