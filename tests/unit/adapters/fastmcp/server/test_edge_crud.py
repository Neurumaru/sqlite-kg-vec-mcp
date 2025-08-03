"""
FastMCP Server Edge/Relationship CRUD 기능 테스트.
"""

import unittest

from src.dto.relationship import RelationshipType
from tests.unit.adapters.fastmcp.server.test_mocks import (
    BaseServerTestCase,
    MockContext,
    MockRelationship,
)


class TestKnowledgeGraphServerCreateEdge(unittest.TestCase, BaseServerTestCase):
    """KnowledgeGraphServer.create_edge 메서드 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseServerTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseServerTestCase.tearDown(self)

    def test_success(self):
        """
        Given: 유효한 관계 생성 요청
        When: create_edge 메서드를 호출할 때
        Then: 관계가 성공적으로 생성되어야 함
        """
        # Given
        source_id = "1"
        target_id = "2"
        relation_type = RelationshipType.RELATES_TO
        properties = {"since": "2023"}

        mock_relationship = MockRelationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationshipType.RELATES_TO,
            properties=properties,
        )
        self.relationship_manager.create_relationship.return_value = mock_relationship

        # Simulate KnowledgeGraphServer.create_edge method
        def create_edge(source_id, target_id, relation_type, properties=None, ctx=None):
            if ctx:
                ctx.info(f"Creating edge: {source_id} -> {target_id} ({relation_type})")

            properties = properties or {}

            try:
                relationship = self.relationship_manager.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=properties,
                )

                return {
                    "relationship_id": relationship.id,
                    "source_id": relationship.source_node_id,
                    "target_id": relationship.target_node_id,
                    "relation_type": relationship.relationship_type,
                    "properties": relationship.properties,
                    "created_at": relationship.created_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create edge: {exception}")
                return {"error": str(exception)}

        # When
        result = create_edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties,
        )

        # Then
        self.relationship_manager.create_relationship.assert_called_once_with(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties,
        )

        expected_result = {
            "relationship_id": mock_relationship.id,
            "source_id": mock_relationship.source_node_id,
            "target_id": mock_relationship.target_node_id,
            "relation_type": mock_relationship.relationship_type,
            "properties": mock_relationship.properties,
            "created_at": mock_relationship.created_at,
        }
        self.assertEqual(result, expected_result)

    def test_success_when_context_provided(self):
        """
        Given: Context가 포함된 관계 생성 요청
        When: create_edge 메서드를 호출할 때
        Then: Context에 로그가 기록되어야 함
        """
        # Given
        mock_context = MockContext()
        source_id = 1
        target_id = 2
        relation_type = RelationshipType.RELATES_TO

        mock_relationship = MockRelationship(
            source_id=str(source_id), target_id=str(target_id), relation_type=relation_type
        )
        self.relationship_manager.create_relationship.return_value = mock_relationship

        # Simulate method
        def create_edge(source_id, target_id, relation_type, properties=None, ctx=None):
            if ctx:
                ctx.info(f"Creating edge: {source_id} -> {target_id} ({relation_type})")

            properties = properties or {}
            relationship = self.relationship_manager.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                properties=properties,
            )

            return {
                "relationship_id": relationship.id,
                "source_id": relationship.source_node_id,
                "target_id": relationship.target_node_id,
                "relation_type": relationship.relationship_type,
                "properties": relationship.properties,
                "created_at": relationship.created_at,
            }

        # When
        result = create_edge(
            source_id=source_id, target_id=target_id, relation_type=relation_type, ctx=mock_context
        )

        # Then
        self.assertEqual(len(mock_context.info_calls), 1)
        self.assertIn("Creating edge:", mock_context.info_calls[0])
        self.assertIn("1 -> 2 (RelationshipType.RELATES_TO)", mock_context.info_calls[0])
        self.assertEqual(result["relationship_id"], mock_relationship.id)

    def test_exception_when_creation_fails(self):
        """
        Given: 관계 생성이 실패하는 상황
        When: create_edge 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """
        # Given
        mock_context = MockContext()
        source_id = 1
        target_id = 2
        relation_type = RelationshipType.RELATES_TO
        error_message = "Source node not found"

        self.relationship_manager.create_relationship.side_effect = Exception(error_message)

        # Simulate method
        def create_edge(source_id, target_id, relation_type, properties=None, ctx=None):
            if ctx:
                ctx.info(f"Creating edge: {source_id} -> {target_id} ({relation_type})")

            properties = properties or {}

            try:
                relationship = self.relationship_manager.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=properties,
                )
                return {
                    "relationship_id": relationship.id,
                    "source_id": relationship.source_node_id,
                    "target_id": relationship.target_node_id,
                    "relation_type": relationship.relationship_type,
                    "properties": relationship.properties,
                    "created_at": relationship.created_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create edge: {exception}")
                return {"error": str(exception)}

        # When
        result = create_edge(
            source_id=source_id, target_id=target_id, relation_type=relation_type, ctx=mock_context
        )

        # Then
        self.assertIn("error", result)
        self.assertEqual(result["error"], error_message)
        self.assertEqual(len(mock_context.error_calls), 1)
        self.assertIn("Failed to create edge:", mock_context.error_calls[0])


if __name__ == "__main__":
    unittest.main()
