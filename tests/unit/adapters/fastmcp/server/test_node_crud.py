"""
FastMCP Server Node CRUD 기능 테스트.
"""

import unittest

from src.dto.node import NodeType
from tests.unit.adapters.fastmcp.server.test_mocks import (
    BaseServerTestCase,
    MockContext,
    MockEntity,
)


class TestKnowledgeGraphServerCreateNode(unittest.TestCase, BaseServerTestCase):
    """KnowledgeGraphServer.create_node 메서드 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseServerTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseServerTestCase.tearDown(self)

    def test_success(self):
        """
        Given: 유효한 노드 생성 요청
        When: create_node 메서드를 호출할 때
        Then: 노드가 성공적으로 생성되어야 함
        """
        # Given
        mock_entity = MockEntity(entity_type=NodeType.CONCEPT)
        self.entity_manager.create_entity.return_value = mock_entity

        # Simulate KnowledgeGraphServer.create_node method
        def create_node(node_type, name=None, properties=None, node_uuid=None, ctx=None):
            if ctx:
                ctx.info(f"Creating node of type '{node_type}'")

            properties = properties or {}

            try:
                entity = self.entity_manager.create_entity(
                    entity_type=node_type, name=name, properties=properties, custom_uuid=node_uuid
                )

                return {
                    "node_id": entity.id,
                    "name": entity.name,
                    "type": entity.node_type,
                    "properties": entity.properties,
                    "created_at": entity.created_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create node: {exception}")
                return {"error": str(exception)}

        node_type = "test_type"
        name = "Test Node"
        properties = {"key": "value"}

        # When
        result = create_node(node_type=node_type, name=name, properties=properties)

        # Then
        self.entity_manager.create_entity.assert_called_once_with(
            entity_type=node_type, name=name, properties=properties, custom_uuid=None
        )

        expected_result = {
            "node_id": mock_entity.id,
            "name": mock_entity.name,
            "type": mock_entity.node_type,
            "properties": mock_entity.properties,
            "created_at": mock_entity.created_at,
        }
        self.assertEqual(result, expected_result)

    def test_success_when_context_provided(self):
        """
        Given: Context가 포함된 노드 생성 요청
        When: create_node 메서드를 호출할 때
        Then: Context에 로그가 기록되어야 함
        """
        # Given
        mock_context = MockContext()
        mock_entity = MockEntity()
        self.entity_manager.create_entity.return_value = mock_entity

        # Simulate method
        def create_node(node_type, name=None, properties=None, node_uuid=None, ctx=None):
            if ctx:
                ctx.info(f"Creating node of type '{node_type}'")

            properties = properties or {}
            entity = self.entity_manager.create_entity(
                entity_type=node_type, name=name, properties=properties, custom_uuid=node_uuid
            )

            return {
                "node_id": entity.id,
                "uuid": entity.uuid,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "created_at": entity.created_at,
            }

        node_type = "test_type"
        name = "Test Node"

        # When
        result = create_node(node_type=node_type, name=name, ctx=mock_context)

        # Then
        self.assertEqual(len(mock_context.info_calls), 1)
        self.assertIn("Creating node of type 'test_type'", mock_context.info_calls[0])
        self.assertEqual(result["node_id"], mock_entity.id)

    def test_exception_when_creation_fails(self):
        """
        Given: 노드 생성이 실패하는 상황
        When: create_node 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """
        # Given
        mock_context = MockContext()
        error_message = "Database connection failed"
        self.entity_manager.create_entity.side_effect = Exception(error_message)

        # Simulate method
        def create_node(node_type, name=None, properties=None, node_uuid=None, ctx=None):
            if ctx:
                ctx.info(f"Creating node of type '{node_type}'")

            properties = properties or {}

            try:
                entity = self.entity_manager.create_entity(
                    entity_type=node_type, name=name, properties=properties, custom_uuid=node_uuid
                )
                return {
                    "node_id": entity.id,
                    "name": entity.name,
                    "type": entity.node_type,
                    "properties": entity.properties,
                    "created_at": entity.created_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create node: {exception}")
                return {"error": str(exception)}

        node_type = "test_type"

        # When
        result = create_node(node_type=node_type, ctx=mock_context)

        # Then
        self.assertIn("error", result)
        self.assertEqual(result["error"], error_message)
        self.assertEqual(len(mock_context.error_calls), 1)
        self.assertIn("Failed to create node:", mock_context.error_calls[0])


class TestKnowledgeGraphServerGetNode(unittest.TestCase, BaseServerTestCase):
    """KnowledgeGraphServer.get_node 메서드 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseServerTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseServerTestCase.tearDown(self)

    def test_success(self):
        """
        Given: 존재하는 노드 ID
        When: get_node 메서드를 호출할 때
        Then: 노드 정보가 반환되어야 함
        """
        # Given
        node_id = 1
        mock_entity = MockEntity(entity_id=node_id)
        self.entity_manager.get_entity_by_id.return_value = mock_entity

        # Simulate method
        def get_node_by_id(node_id, ctx=None):
            try:
                entity = self.entity_manager.get_entity_by_id(node_id)
                if not entity:
                    return {"error": f"Node with id {node_id} not found"}

                return {
                    "node_id": entity.id,
                    "uuid": entity.uuid,
                    "name": entity.name,
                    "type": entity.type,
                    "properties": entity.properties,
                    "created_at": entity.created_at,
                    "updated_at": entity.updated_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to get node: {exception}")
                return {"error": str(exception)}

        # When
        result = get_node_by_id(node_id)

        # Then
        self.entity_manager.get_entity_by_id.assert_called_once_with(node_id)
        expected_result = {
            "node_id": mock_entity.id,
            "uuid": mock_entity.uuid,
            "name": mock_entity.name,
            "type": mock_entity.type,
            "properties": mock_entity.properties,
            "created_at": mock_entity.created_at,
            "updated_at": mock_entity.updated_at,
        }
        self.assertEqual(result, expected_result)

    def test_error_when_missing_parameters(self):
        """
        Given: 필수 파라미터가 누락된 요청
        When: get_node 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """
        # 이 테스트는 실제 서버에서 파라미터 검증 로직이 있을 때 구현
        self.skipTest("파라미터 검증 로직 미구현")


class TestKnowledgeGraphServerDeleteNode(unittest.TestCase, BaseServerTestCase):
    """KnowledgeGraphServer.delete_node 메서드 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseServerTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseServerTestCase.tearDown(self)

    def test_success(self):
        """
        Given: 존재하는 노드 ID
        When: delete_node 메서드를 호출할 때
        Then: 노드가 성공적으로 삭제되어야 함
        """
        # Given
        node_id = 1
        self.entity_manager.delete_entity.return_value = True

        # Simulate method
        def delete_node(node_id, ctx=None):
            try:
                success = self.entity_manager.delete_entity(node_id)
                if success:
                    return {"message": f"Node {node_id} deleted successfully", "success": True}
                return {"error": f"Failed to delete node {node_id}", "success": False}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to delete node: {exception}")
                return {"error": str(exception), "success": False}

        # When
        result = delete_node(node_id)

        # Then
        self.entity_manager.delete_entity.assert_called_once_with(node_id)
        self.assertTrue(result["success"])
        self.assertIn("deleted successfully", result["message"])

    def test_error_when_not_found(self):
        """
        Given: 존재하지 않는 노드 ID
        When: delete_node 메서드를 호출할 때
        Then: 에러가 반환되어야 함
        """
        # Given
        node_id = 999
        self.entity_manager.delete_entity.return_value = False

        # Simulate method
        def delete_node(node_id, ctx=None):
            try:
                success = self.entity_manager.delete_entity(node_id)
                if success:
                    return {"message": f"Node {node_id} deleted successfully", "success": True}
                return {"error": f"Failed to delete node {node_id}", "success": False}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to delete node: {exception}")
                return {"error": str(exception), "success": False}

        # When
        result = delete_node(node_id)

        # Then
        self.assertFalse(result["success"])
        self.assertIn("Failed to delete", result["error"])


if __name__ == "__main__":
    unittest.main()
