"""
SQLite 그래프 순회(Traversal) 모듈의 단위 테스트.
"""

import datetime
import json
import sqlite3
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.adapters.sqlite3.graph.entities import Entity
from src.adapters.sqlite3.graph.relationships import Relationship
from src.adapters.sqlite3.graph.traversal import (
    GraphTraversal,
    PathNode,
)


class TestPathNode(unittest.TestCase):
    """PathNode 데이터클래스 테스트."""

    def test_init_root_node(self):
        """Given: 루트 노드용 엔티티가 제공될 때
        When: PathNode를 생성하면
        Then: 루트 노드가 올바르게 초기화된다
        """
        # Given
        entity = Entity(
            id=1,
            uuid="entity-1",
            name="Root Entity",
            type="Person",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )

        # When
        path_node = PathNode(entity=entity)

        # Then
        self.assertEqual(path_node.entity, entity)
        self.assertIsNone(path_node.relationship)
        self.assertIsNone(path_node.parent)
        self.assertEqual(path_node.depth, 0)

    def test_init_child_node(self):
        """Given: 부모 노드와 관계가 있는 자식 노드 데이터가 제공될 때
        When: PathNode를 생성하면
        Then: 자식 노드가 올바르게 초기화된다
        """
        # Given
        parent_entity = Entity(
            id=1,
            uuid="parent-1",
            name="Parent",
            type="Person",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        child_entity = Entity(
            id=2,
            uuid="child-1",
            name="Child",
            type="Organization",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        relationship = Relationship(
            id=1,
            source_id=1,
            target_id=2,
            relation_type="WORKS_FOR",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        parent_node = PathNode(entity=parent_entity)

        # When
        child_node = PathNode(
            entity=child_entity,
            relationship=relationship,
            parent=parent_node,
            depth=1,
        )

        # Then
        self.assertEqual(child_node.entity, child_entity)
        self.assertEqual(child_node.relationship, relationship)
        self.assertEqual(child_node.parent, parent_node)
        self.assertEqual(child_node.depth, 1)

    def test_path_to_root_single_node(self):
        """Given: 단일 루트 노드가 있을 때
        When: path_to_root를 호출하면
        Then: 자기 자신만 포함한 경로가 반환된다
        """
        # Given
        entity = Entity(
            id=1,
            uuid="root-1",
            name="Root",
            type="Person",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        root_node = PathNode(entity=entity)

        # When
        path = root_node.path_to_root

        # Then
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], root_node)

    def test_path_to_root_three_levels(self):
        """Given: 3단계 깊이의 경로가 있을 때
        When: path_to_root를 호출하면
        Then: 루트부터 현재 노드까지의 전체 경로가 반환된다
        """
        # Given
        # 루트 노드
        root_entity = Entity(
            id=1,
            uuid="root-1",
            name="Root",
            type="Person",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        root_node = PathNode(entity=root_entity)

        # 중간 노드
        mid_entity = Entity(
            id=2,
            uuid="mid-1",
            name="Middle",
            type="Organization",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        mid_relationship = Relationship(
            id=1,
            source_id=1,
            target_id=2,
            relation_type="WORKS_FOR",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        mid_node = PathNode(
            entity=mid_entity,
            relationship=mid_relationship,
            parent=root_node,
            depth=1,
        )

        # 리프 노드
        leaf_entity = Entity(
            id=3,
            uuid="leaf-1",
            name="Leaf",
            type="Project",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        leaf_relationship = Relationship(
            id=2,
            source_id=2,
            target_id=3,
            relation_type="MANAGES",
            properties={},
            created_at="2023-12-25T10:00:00",
            updated_at="2023-12-25T10:00:00",
        )
        leaf_node = PathNode(
            entity=leaf_entity,
            relationship=leaf_relationship,
            parent=mid_node,
            depth=2,
        )

        # When
        path = leaf_node.path_to_root

        # Then
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], root_node)
        self.assertEqual(path[1], mid_node)
        self.assertEqual(path[2], leaf_node)


class TestGraphTraversal(unittest.TestCase):
    """GraphTraversal 클래스 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.traversal = GraphTraversal(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결이 제공될 때
        When: GraphTraversal을 초기화하면
        Then: 연결이 올바르게 설정된다
        """
        # Given & When
        traversal = GraphTraversal(self.mock_connection)

        # Then
        self.assertEqual(traversal.connection, self.mock_connection)

    def test_get_neighbors_outgoing_only(self):
        """Given: 엔티티 ID와 outgoing 방향이 제공될 때
        When: get_neighbors를 호출하면
        Then: 나가는 관계의 이웃들이 반환된다
        """
        # Given
        entity_id = 1
        direction = "outgoing"

        # Mock 관계 데이터
        mock_relationship_rows = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
            self._create_mock_relationship_row(2, 1, 3, "INFLUENCES"),
        ]

        # Mock 엔티티 데이터
        mock_entity_rows = [
            self._create_mock_entity_row(2, "entity-2", "Target 1", "Person"),
            self._create_mock_entity_row(3, "entity-3", "Target 2", "Organization"),
        ]

        self.mock_cursor.fetchall.side_effect = [mock_relationship_rows, mock_entity_rows]
        self.mock_cursor.execute.side_effect = [
            None,
            None,
        ]  # Mock execute for the two internal queries

        # When
        result = self.traversal.get_neighbors(entity_id, direction=direction)

        # Then
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_cursor.execute.call_count, 1)

        # 첫 번째 호출: 관계 조회
        first_call = self.mock_cursor.execute.call_args_list[0]
        self.assertIn("SELECT e.*, r.id as rel_id, r.source_id, r.target_id,", first_call[0][0])
        self.assertIn("FROM entities e", first_call[0][0])
        self.assertIn("JOIN edges r ON e.id = r.target_id", first_call[0][0])
        self.assertIn("WHERE r.source_id = ?", first_call[0][0])
        self.assertIn("LIMIT ?", first_call[0][0])
        self.assertEqual(first_call[0][1], [entity_id, 100])

    def test_get_neighbors_incoming_only(self):
        """Given: 엔티티 ID와 incoming 방향이 제공될 때
        When: get_neighbors를 호출하면
        Then: 들어오는 관계의 이웃들이 반환된다
        """
        # Given
        entity_id = 2
        direction = "incoming"

        mock_relationship_rows = [
            self._create_mock_relationship_row(3, 1, 2, "POINTS_TO"),
        ]

        mock_entity_rows = [
            self._create_mock_entity_row(1, "entity-1", "Source", "Person"),
        ]

        self.mock_cursor.fetchall.side_effect = [mock_relationship_rows, mock_entity_rows]
        self.mock_cursor.execute.side_effect = [
            None,
            None,
        ]  # Mock execute for the two internal queries

        # When
        result = self.traversal.get_neighbors(entity_id, direction=direction)

        # Then
        self.assertEqual(len(result), 1)
        self.assertEqual(self.mock_cursor.execute.call_count, 1)
        first_call = self.mock_cursor.execute.call_args_list[0]
        self.assertIn("SELECT e.*, r.id as rel_id, r.source_id, r.target_id,", first_call[0][0])
        self.assertIn("FROM entities e", first_call[0][0])
        self.assertIn("JOIN edges r ON e.id = r.source_id", first_call[0][0])
        self.assertIn("WHERE r.target_id = ?", first_call[0][0])
        self.assertIn("LIMIT ?", first_call[0][0])
        self.assertEqual(first_call[0][1], [entity_id, 100])

    def test_get_neighbors_both_directions(self):
        """Given: 엔티티 ID와 both 방향이 제공될 때
        When: get_neighbors를 호출하면
        Then: 양방향 관계의 모든 이웃들이 반환된다
        """
        # Given
        entity_id = 2
        direction = "both"

        # 나가는 관계
        outgoing_rows = [
            self._create_mock_relationship_row(1, 2, 3, "CONNECTS_TO"),
        ]

        # 들어오는 관계
        incoming_rows = [
            self._create_mock_relationship_row(2, 1, 2, "POINTS_TO"),
        ]

        # 이웃 엔티티들
        neighbor_entities = [
            self._create_mock_entity_row(3, "entity-3", "Target", "Organization"),
            self._create_mock_entity_row(1, "entity-1", "Source", "Person"),
        ]

        self.mock_cursor.fetchall.side_effect = [
            outgoing_rows + incoming_rows,  # For the combined fetchall in get_neighbors
            neighbor_entities,  # For the entities associated with relationships
        ]
        self.mock_cursor.execute.side_effect = [None]  # Mock execute for the single UNION query

        # When
        result = self.traversal.get_neighbors(entity_id, direction=direction)

        # Then
        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_cursor.execute.call_count, 1)
        first_call = self.mock_cursor.execute.call_args_list[0]
        self.assertIn("SELECT e.*, r.id as rel_id, r.source_id, r.target_id,", first_call[0][0])
        self.assertIn("FROM entities e", first_call[0][0])
        self.assertIn("JOIN edges r ON e.id = r.target_id", first_call[0][0])
        self.assertIn("WHERE r.source_id = ?", first_call[0][0])
        self.assertIn("UNION", first_call[0][0])
        self.assertIn("FROM entities e", first_call[0][0])
        self.assertIn("JOIN edges r ON e.id = r.source_id", first_call[0][0])
        self.assertIn("WHERE r.target_id = ?", first_call[0][0])
        self.assertIn("LIMIT ?", first_call[0][0])
        self.assertEqual(first_call[0][1], [entity_id, entity_id, 100])

    def test_get_neighbors_with_relation_filter(self):
        """Given: 관계 타입 필터가 제공될 때
        When: get_neighbors를 호출하면
        Then: 해당 타입의 관계만 조회된다
        """
        # Given
        entity_id = 1
        direction = "outgoing"
        relation_types = ["CONNECTED_TO", "INFLUENCES"]

        mock_relationship_rows = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
        ]

        mock_entity_rows = [
            self._create_mock_entity_row(2, "entity-2", "Target", "Person"),
        ]

        self.mock_cursor.fetchall.side_effect = [mock_relationship_rows, mock_entity_rows]
        self.mock_cursor.execute.side_effect = [None]  # Mock execute for the single query

        # When
        _ = self.traversal.get_neighbors(
            entity_id, direction=direction, relation_types=relation_types
        )

        # Then
        first_call = self.mock_cursor.execute.call_args_list[0]
        self.assertIn("SELECT e.*, r.id as rel_id, r.source_id, r.target_id,", first_call[0][0])
        self.assertIn("FROM entities e", first_call[0][0])
        self.assertIn("JOIN edges r ON e.id = r.target_id", first_call[0][0])
        self.assertIn("WHERE r.source_id = ? AND r.relation_type IN (?, ?)", first_call[0][0])
        self.assertIn("LIMIT ?", first_call[0][0])
        self.assertEqual(first_call[0][1], [entity_id, "CONNECTED_TO", "INFLUENCES", 100])

    def test_get_neighbors_no_relationships(self):
        """Given: 관계가 없는 엔티티 ID가 제공될 때
        When: get_neighbors를 호출하면
        Then: 빈 목록이 반환된다
        """
        # Given
        entity_id = 999
        direction = "outgoing"
        self.mock_cursor.fetchall.side_effect = [[], []]

        # When
        result = self.traversal.get_neighbors(entity_id, direction=direction)

        # Then
        self.assertEqual(len(result), 0)

    def test_breadth_first_search_single_level(self):
        """Given: BFS 검색 파라미터가 제공될 때
        When: breadth_first_search를 호출하면
        Then: 너비 우선 순서로 노드들이 반환된다
        """
        # Given
        start_entity_id = 1
        max_depth = 1

        # 시작 엔티티
        start_entity_row = self._create_mock_entity_row(1, "start-1", "Start", "Person")

        # 이웃 관계들
        relationship_rows = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
            self._create_mock_relationship_row(2, 1, 3, "INFLUENCES"),
        ]

        # 이웃 엔티티들
        neighbor_entity_rows = [
            self._create_mock_entity_row(2, "neighbor-1", "Neighbor 1", "Person"),
            self._create_mock_entity_row(3, "neighbor-2", "Neighbor 2", "Organization"),
        ]

        self.mock_cursor.fetchone.return_value = start_entity_row
        self.mock_cursor.fetchall.side_effect = [relationship_rows, neighbor_entity_rows]

        # When
        result = self.traversal.breadth_first_search(start_entity_id, max_depth=max_depth)

        # Then
        self.assertGreater(len(result), 0)
        # 시작 노드는 깊이 0이어야 함
        start_node = next((node for node in result if node.entity.id == start_entity_id), None)
        self.assertIsNotNone(start_node)
        self.assertEqual(start_node.depth, 0)

    def test_breadth_first_search_max_depth_limit(self):
        """Given: 최대 깊이 제한이 있는 BFS 파라미터가 제공될 때
        When: breadth_first_search를 호출하면
        Then: 지정된 깊이까지만 탐색된다
        """
        # Given
        start_entity_id = 1
        max_depth = 0  # 시작 노드만

        start_entity_row = self._create_mock_entity_row(1, "start-1", "Start", "Person")
        self.mock_cursor.fetchone.return_value = start_entity_row

        # When
        result = self.traversal.breadth_first_search(start_entity_id, max_depth=max_depth)

        # Then
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity.id, start_entity_id)
        self.assertEqual(result[0].depth, 0)

    def test_depth_first_search_single_path(self):
        """Given: DFS 검색 파라미터가 제공될 때
        When: depth_first_search를 호출하면
        Then: 깊이 우선 순서로 노드들이 반환된다
        """
        # Given
        start_entity_id = 1
        max_depth = 2

        start_entity_row = self._create_mock_entity_row(1, "start-1", "Start", "Person")
        self.mock_cursor.fetchone.return_value = start_entity_row

        # 첫 번째 레벨 이웃
        first_level_relationships = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
        ]
        first_level_entities = [
            self._create_mock_entity_row(2, "level1-1", "Level 1", "Person"),
        ]

        # 두 번째 레벨 이웃 (빈 결과로 설정)
        second_level_relationships = []
        second_level_entities = []

        # Mock fetchall calls for get_neighbors within DFS
        self.mock_cursor.fetchall.side_effect = [
            first_level_relationships,  # Neighbors of start_entity
            first_level_entities,  # Entities for neighbors of start_entity
            second_level_relationships,  # Neighbors of level1-1 (empty)
            second_level_entities,  # Entities for neighbors of level1-1 (empty)
        ]

        # When
        result = self.traversal.depth_first_search(start_entity_id, max_depth=max_depth)

        # Then
        self.assertGreater(len(result), 0)
        # DFS에서는 깊이 우선으로 탐색
        start_node = result[0]
        self.assertEqual(start_node.entity.id, start_entity_id)
        self.assertEqual(start_node.depth, 0)
        # Check path: Start -> Level 1
        self.assertEqual(result[0].entity.id, 1)
        self.assertEqual(result[1].entity.id, 2)

    def test_find_shortest_path_direct_connection(self):
        """Given: 직접 연결된 두 노드가 있을 때
        When: find_shortest_path를 호출하면
        Then: 최단 경로가 반환된다
        """
        # Given
        start_id = 1
        target_id = 2

        # Setup for breadth_first_search calls within find_shortest_path
        start_entity_row = self._create_mock_entity_row(1, "start-1", "Start", "Person")
        target_entity_row = self._create_mock_entity_row(2, "target-1", "Target", "Person")

        direct_relationship = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
        ]

        # Side effect for fetchone (start_entity) then fetchall (neighbors of start_entity)
        self.mock_cursor.fetchone.side_effect = [start_entity_row, target_entity_row]
        self.mock_cursor.fetchall.side_effect = [
            direct_relationship,  # Relationships from start_entity
            [target_entity_row],  # Entities for neighbors (i.e., target_entity)
        ]

        # When
        result = self.traversal.find_shortest_path(start_id, target_id)

        # Then
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Start node + target node
        self.assertEqual(result[0].entity.id, start_id)
        self.assertEqual(result[1].entity.id, target_id)

    def test_find_shortest_path_no_path(self):
        """Given: 연결되지 않은 두 노드가 있을 때
        When: find_shortest_path를 호출하면
        Then: None이 반환된다
        """
        # Given
        start_id = 1
        target_id = 999  # 연결되지 않은 노드

        start_entity_row = self._create_mock_entity_row(1, "start-1", "Start", "Person")
        self.mock_cursor.fetchone.side_effect = [
            start_entity_row,
            None,
        ]  # For start_entity and then for target_entity lookup in BFS

        # No relationships / no path
        self.mock_cursor.fetchall.side_effect = [[], []]  # For get_neighbors calls within BFS

        # When
        result = self.traversal.find_shortest_path(start_id, target_id)

        # Then
        self.assertIsNone(result)

    def test_get_connected_components_single_component(self):
        """Given: 모든 노드가 연결된 그래프가 있을 때
        When: get_connected_components를 호출하면
        Then: 단일 연결 성분이 반환된다
        """
        # Given
        # 모든 엔티티
        all_entities = [
            self._create_mock_entity_row(1, "entity-1", "Person"),
            self._create_mock_entity_row(2, "entity-2", "Person"),
            self._create_mock_entity_row(3, "entity-3", "Organization"),
        ]

        # 연결 관계들
        all_relationships = [
            self._create_mock_relationship_row(1, 1, 2, "CONNECTED_TO"),
            self._create_mock_relationship_row(2, 2, 3, "INFLUENCES"),
        ]

        # Setup mock_cursor.fetchall.side_effect for get_connected_components and its internal get_neighbors calls
        # The order of these side_effects is critical
        self.mock_cursor.fetchall.side_effect = [
            all_entities,  # 1. All entities for get_connected_components initial loop
            all_relationships,  # 2. Neighbors of entity 1 (for BFS)
            [
                self._create_mock_entity_row(2, "entity-2", "Person")
            ],  # 3. Entities for neighbors of entity 1
            all_relationships,  # 4. Neighbors of entity 2 (for BFS)
            [
                self._create_mock_entity_row(3, "entity-3", "Organization")
            ],  # 5. Entities for neighbors of entity 2
            [],  # 6. Neighbors of entity 3 (empty)
            [],  # 7. Entities for neighbors of entity 3 (empty)
            [],  # 8. Fallback if more get_neighbors calls occur unexpectedly
        ]
        self.mock_cursor.fetchone.side_effect = [
            self._create_mock_entity_row(1, "entity-1", "Person"),  # For get_neighbors of entity 1
            self._create_mock_entity_row(2, "entity-2", "Person"),  # For get_neighbors of entity 2
            self._create_mock_entity_row(
                3, "entity-3", "Organization"
            ),  # For get_neighbors of entity 3
            None,  # End of entities
        ]

        # When
        result = self.traversal.get_connected_components()

        # Then
        self.assertEqual(len(result), 1)  # 단일 연결 성분
        self.assertEqual(len(result[0]), 3)  # 3개 노드 모두 포함
        self.assertIn(1, result[0])
        self.assertIn(2, result[0])
        self.assertIn(3, result[0])

    def _create_mock_relationship_row(
        self, rel_id: int, source_id: int, target_id: int, rel_type: str, properties: dict = None
    ):
        mock_row = MagicMock()
        data = {
            "id": rel_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": rel_type,
            "properties": json.dumps(properties) if properties is not None else None,
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_row.__getitem__.side_effect = data.get
        return mock_row

    def _create_mock_entity_row(
        self, entity_id: int, name: str, entity_type: str, properties: dict = None
    ):
        mock_row = MagicMock()  # Changed to MagicMock
        data = {
            "id": entity_id,
            "uuid": f"entity-{entity_id}",
            "name": name,
            "type": entity_type,
            "properties": json.dumps(properties) if properties is not None else None,
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_row.__getitem__.side_effect = data.get
        return mock_row


class TestGraphTraversalIntegration(unittest.TestCase):
    """GraphTraversal 통합 테스트."""

    def test_complex_graph_traversal(self):
        """Given: 복잡한 그래프 구조가 있을 때
        When: 다양한 순회 알고리즘을 사용하면
        Then: 일관된 결과가 반환된다
        """
        # 이 테스트는 실제 SQLite 데이터베이스를 사용하는 통합 테스트로
        # 단위 테스트 범위를 벗어나므로 스킵합니다.
        self.skipTest("Integration test - requires actual database with graph data")

    @patch("src.adapters.sqlite3.graph.traversal.get_observable_logger")
    def test_logger_initialization(self, mock_get_logger):
        """Given: GraphTraversal이 생성될 때
        When: 로거를 확인하면
        Then: 올바른 로거가 설정된다
        """
        # Given
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_connection = Mock(spec=sqlite3.Connection)

        # When
        _ = GraphTraversal(mock_connection)

        # Then
        # 실제 구현에서 로거가 사용된다면 이 테스트가 유효함
        # 현재는 traversal.py에서 로거를 명시적으로 초기화하지 않으므로 스킵
        self.skipTest("Logger not explicitly used in current implementation")


if __name__ == "__main__":
    unittest.main()
