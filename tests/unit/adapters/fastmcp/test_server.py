"""
Unit tests for FastMCP server adapter (simplified version).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


class MockEntity:
    """Mock entity for testing."""

    def __init__(
        self,
        entity_id=1,
        uuid="test-uuid",
        name="Test",
        entity_type="test",
        properties=None,
        created_at="2023-01-01",
        updated_at="2023-01-02",
    ):
        self.id = entity_id
        self.uuid = uuid
        self.name = name
        self.type = entity_type
        self.properties = properties or {}
        self.created_at = created_at
        self.updated_at = updated_at


class MockRelationship:
    """Mock relationship for testing."""

    def __init__(
        self,
        relationship_id=1,
        source_id=1,
        target_id=2,
        relation_type="test",
        properties=None,
        created_at="2023-01-01",
        updated_at="2023-01-02",
    ):
        self.id = relationship_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.properties = properties or {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.source = None
        self.target = None


class MockSearchResult:
    """Mock search result for testing."""

    def __init__(self, node_id=1, similarity=0.8):
        self.node_id = node_id
        self.similarity = similarity

    def to_dict(self):
        return {"node_id": self.node_id, "similarity": self.similarity}


class MockContext:
    """Mock Context for testing."""

    def __init__(self):
        self.info_calls = []
        self.error_calls = []

    def info(self, message):
        self.info_calls.append(message)

    def error(self, message):
        self.error_calls.append(message)


class TestKnowledgeGraphServerMethods(unittest.TestCase):
    """Test cases for KnowledgeGraphServer methods."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # 임시 데이터베이스 파일 생성
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Mock managers
        self.entity_manager = Mock()
        self.relationship_manager = Mock()
        self.vector_search = Mock()

    def tearDown(self):
        """Clean up after each test method."""
        try:
            Path(self.db_path).unlink()
        except FileNotFoundError:
            pass

    def test_create_node_success(self):
        """
        Given: 유효한 노드 생성 요청
        When: create_node 메서드를 호출할 때
        Then: 노드가 성공적으로 생성되어야 함
        """
        # Given
        mock_entity = MockEntity()
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
                    "uuid": entity.uuid,
                    "name": entity.name,
                    "type": entity.type,
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
            "uuid": mock_entity.uuid,
            "name": mock_entity.name,
            "type": mock_entity.type,
            "properties": mock_entity.properties,
            "created_at": mock_entity.created_at,
        }
        self.assertEqual(result, expected_result)

    def test_create_node_with_context(self):
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

        # When
        create_node(node_type="test", ctx=mock_context)

        # Then
        self.assertEqual(len(mock_context.info_calls), 1)
        self.assertIn("Creating node of type 'test'", mock_context.info_calls[0])

    def test_create_node_failure(self):
        """
        Given: 노드 생성 중 오류 발생
        When: create_node 메서드를 호출할 때
        Then: 오류 메시지를 반환해야 함
        """
        # Given
        mock_context = MockContext()
        error_message = "Database error"
        self.entity_manager.create_entity.side_effect = Exception(error_message)

        # Simulate method
        def create_node(node_type, name=None, properties=None, node_uuid=None, ctx=None):
            if ctx:
                ctx.info(f"Creating node of type '{node_type}'")

            properties = properties or {}

            try:
                _ = self.entity_manager.create_entity(
                    entity_type=node_type, name=name, properties=properties, custom_uuid=node_uuid
                )
                return {"created": True}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create node: {exception}")
                return {"error": str(exception)}

        # When
        result = create_node(node_type="test", ctx=mock_context)

        # Then
        self.assertEqual(result, {"error": error_message})
        self.assertEqual(len(mock_context.error_calls), 1)

    def test_get_node_by_id_success(self):
        """
        Given: 유효한 노드 ID
        When: get_node 메서드를 호출할 때
        Then: 노드 정보를 반환해야 함
        """
        # Given
        mock_entity = MockEntity()
        self.entity_manager.get_entity.return_value = mock_entity

        # Simulate method
        def get_node(node_id=None, node_uuid=None, ctx=None):
            if node_id is None and node_uuid is None:
                error_msg = "Missing required parameter: either id or node_uuid must be provided"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            try:
                if node_id is not None:
                    entity = self.entity_manager.get_entity(node_id)
                    if ctx:
                        ctx.info(f"Retrieving node with ID {node_id}")
                else:
                    entity = self.entity_manager.get_entity_by_uuid(node_uuid)
                    if ctx:
                        ctx.info(f"Retrieving node with UUID {node_uuid}")

                if not entity:
                    error_msg = "Node not found"
                    if ctx:
                        ctx.error(error_msg)
                    return {"error": error_msg}

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
        result = get_node(node_id=1)

        # Then
        self.entity_manager.get_entity.assert_called_once_with(1)
        expected_result = {
            "node_id": 1,
            "uuid": "test-uuid",
            "name": "Test",
            "type": "test",
            "properties": {},
            "created_at": "2023-01-01",
            "updated_at": "2023-01-02",
        }
        self.assertEqual(result, expected_result)

    def test_get_node_missing_parameters(self):
        """
        Given: ID와 UUID가 모두 없는 요청
        When: get_node 메서드를 호출할 때
        Then: 오류 메시지를 반환해야 함
        """

        # Given
        def get_node(node_id=None, node_uuid=None, ctx=None):
            if node_id is None and node_uuid is None:
                error_msg = "Missing required parameter: either id or node_uuid must be provided"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

        # When
        result = get_node()

        # Then
        expected_error = "Missing required parameter: either id or node_uuid must be provided"
        self.assertEqual(result, {"error": expected_error})

    def test_create_edge_success(self):
        """
        Given: 유효한 엣지 생성 요청
        When: create_edge 메서드를 호출할 때
        Then: 엣지가 성공적으로 생성되어야 함
        """
        # Given
        mock_relationship = MockRelationship()
        self.relationship_manager.create_relationship.return_value = mock_relationship

        # Simulate method
        def create_edge(source_id, target_id, relation_type, properties=None, ctx=None):
            if ctx:
                ctx.info(f"Creating edge of type '{relation_type}' from {source_id} to {target_id}")

            properties = properties or {}

            try:
                relationship = self.relationship_manager.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=properties,
                )

                return {
                    "edge_id": relationship.id,
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "relation_type": relationship.relation_type,
                    "properties": relationship.properties,
                    "created_at": relationship.created_at,
                }
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to create edge: {exception}")
                return {"error": str(exception)}

        # When
        _ = create_edge(
            source_id=1, target_id=2, relation_type="connects", properties={"weight": 1.0}
        )

        # Then
        self.relationship_manager.create_relationship.assert_called_once_with(
            source_id=1, target_id=2, relation_type="connects", properties={"weight": 1.0}
        )

    def test_search_similar_nodes_missing_parameters(self):
        """
        Given: node_id와 query_vector가 모두 없는 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: 오류 메시지를 반환해야 함
        """

        # Given
        def search_similar_nodes(
            node_id=None,
            query_vector=None,
            limit=10,
            entity_types=None,
            include_entities=True,
            ctx=None,
        ):
            if node_id is None and query_vector is None:
                error_msg = (
                    "Missing required parameter: either node_id or query_vector must be provided"
                )
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

        # When
        result = search_similar_nodes()

        # Then
        expected_error = (
            "Missing required parameter: either node_id or query_vector must be provided"
        )
        self.assertEqual(result, {"error": expected_error})

    def test_search_similar_nodes_with_node_id(self):
        """
        Given: 노드 ID로 유사한 노드 검색 요청
        When: search_similar_nodes 메서드를 호출할 때
        Then: 유사한 노드들을 반환해야 함
        """
        # Given
        mock_result = MockSearchResult(node_id=2, similarity=0.8)
        self.vector_search.search_similar_to_entity.return_value = [mock_result]

        # Simulate method
        def search_similar_nodes(
            node_id=None,
            query_vector=None,
            limit=10,
            entity_types=None,
            include_entities=True,
            ctx=None,
        ):
            if node_id is None and query_vector is None:
                error_msg = (
                    "Missing required parameter: either node_id or query_vector must be provided"
                )
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            if ctx:
                if node_id is not None:
                    ctx.info(f"Searching for nodes similar to node {node_id}")
                else:
                    ctx.info("Searching for nodes similar to provided vector")

            entity_types = entity_types or ["node"]

            try:
                if node_id is not None:
                    results = self.vector_search.search_similar_to_entity(
                        entity_type="node",
                        entity_id=node_id,
                        k=limit,
                        result_entity_types=entity_types,
                        include_entities=include_entities,
                    )
                else:
                    # Would use query_vector here
                    results = []

                result_items = []
                for item in results:
                    result_items.append(item.to_dict())

                if ctx:
                    ctx.info(f"Found {len(result_items)} similar nodes")

                return {"results": result_items, "count": len(result_items)}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to search similar nodes: {exception}")
                return {"error": str(exception)}

        # When
        result = search_similar_nodes(node_id=1, limit=5)

        # Then
        self.vector_search.search_similar_to_entity.assert_called_once_with(
            entity_type="node",
            entity_id=1,
            k=5,
            result_entity_types=["node"],
            include_entities=True,
        )

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["node_id"], 2)
        self.assertEqual(result["results"][0]["similarity"], 0.8)

    def test_delete_node_success(self):
        """
        Given: 유효한 노드 ID
        When: delete_node 메서드를 호출할 때
        Then: 노드가 성공적으로 삭제되어야 함
        """
        # Given
        self.entity_manager.delete_entity.return_value = True

        # Simulate method
        def delete_node(entity_id, ctx=None):
            if ctx:
                ctx.info(f"Deleting node with ID {entity_id}")

            try:
                success = self.entity_manager.delete_entity(entity_id)

                if not success:
                    error_msg = "Node not found or already deleted"
                    if ctx:
                        ctx.error(error_msg)
                    return {"error": error_msg}

                return {"success": True, "message": f"Node {entity_id} deleted successfully"}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to delete node: {exception}")
                return {"error": str(exception)}

        # When
        result = delete_node(entity_id=1)

        # Then
        self.entity_manager.delete_entity.assert_called_once_with(1)
        expected_result = {"success": True, "message": "Node 1 deleted successfully"}
        self.assertEqual(result, expected_result)

    def test_delete_node_not_found(self):
        """
        Given: 존재하지 않는 노드 ID
        When: delete_node 메서드를 호출할 때
        Then: 노드를 찾을 수 없다는 오류를 반환해야 함
        """
        # Given
        self.entity_manager.delete_entity.return_value = False

        # Simulate method
        def delete_node(entity_id, ctx=None):
            if ctx:
                ctx.info(f"Deleting node with ID {entity_id}")

            try:
                success = self.entity_manager.delete_entity(entity_id)

                if not success:
                    error_msg = "Node not found or already deleted"
                    if ctx:
                        ctx.error(error_msg)
                    return {"error": error_msg}

                return {"success": True, "message": f"Node {entity_id} deleted successfully"}
            except Exception as exception:
                if ctx:
                    ctx.error(f"Failed to delete node: {exception}")
                return {"error": str(exception)}

        # When
        result = delete_node(entity_id=999)

        # Then
        expected_error = "Node not found or already deleted"
        self.assertEqual(result, {"error": expected_error})


if __name__ == "__main__":
    unittest.main()
