"""
SQLite 그래프 RelationshipManager 조회 및 검색 기능 테스트.
"""

import datetime
import json
import sqlite3
import unittest
from unittest.mock import MagicMock, Mock

from src.adapters.sqlite3.graph.relationships import (
    Relationship,
    RelationshipManager,
)


class TestRelationshipManagerQueries(unittest.TestCase):
    """RelationshipManager 조회 및 검색 기능 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager = RelationshipManager(self.mock_connection)

    def test_success_when_get_relationships_by_source(self):
        """Given: 소스 엔티티 ID가 제공될 때
        When: get_entity_relationships를 호출하면
        Then: 해당 소스의 모든 관계가 반환된다
        """
        # Given
        source_id = 10
        mock_rows = [
            self._create_mock_row(1, source_id, 20, "CONNECTED_TO"),
            self._create_mock_row(2, source_id, 30, "RELATES_TO"),
        ]
        self.mock_cursor.fetchall.side_effect = [
            mock_rows,  # First call for relationships
            [
                self._create_mock_row_for_entity(source_id, "Source", "Person")
            ],  # For source entity in _load_relationship_entities
            [
                self._create_mock_row_for_entity(20, "Target", "Location")
            ],  # For target entity in _load_relationship_entities
            [
                self._create_mock_row_for_entity(source_id, "Source", "Person")
            ],  # For source entity in _load_relationship_entities
            [
                self._create_mock_row_for_entity(30, "Target", "Location")
            ],  # For target entity in _load_relationship_entities
        ]
        self.mock_cursor.fetchone.return_value = (2,)  # total_count

        # When
        result, total_count = self.manager.get_entity_relationships(
            entity_id=source_id, direction="outgoing"
        )

        # Then
        self.assertEqual(len(result), 2)
        self.assertEqual(total_count, 2)
        self.assertIsInstance(result[0], Relationship)
        self.assertIsInstance(result[1], Relationship)
        self.assertEqual(result[0].source_id, source_id)
        self.assertEqual(result[1].source_id, source_id)

    def test_success_when_get_relationships_by_target(self):
        """Given: 타겟 엔티티 ID가 제공될 때
        When: get_entity_relationships를 호출하면
        Then: 해당 타겟의 모든 관계가 반환된다
        """
        # Given
        target_id = 20
        mock_rows = [
            self._create_mock_row(3, 10, target_id, "INFLUENCES"),
            self._create_mock_row(4, 15, target_id, "CONTAINS"),
        ]
        self.mock_cursor.fetchall.side_effect = [
            mock_rows,  # First call for relationships
            [self._create_mock_row_for_entity(10, "Source", "Person")],  # For source entity 10
            [
                self._create_mock_row_for_entity(target_id, "Target", "Location")
            ],  # For target entity 20
            [self._create_mock_row_for_entity(15, "Source2", "Person")],  # For source entity 15
            [
                self._create_mock_row_for_entity(target_id, "Target2", "Location")
            ],  # For target entity 20
        ]
        self.mock_cursor.fetchone.return_value = (2,)  # total_count

        # When
        result, total_count = self.manager.get_entity_relationships(
            entity_id=target_id, direction="incoming"
        )

        # Then
        self.assertEqual(len(result), 2)
        self.assertEqual(total_count, 2)
        self.assertEqual(result[0].target_id, target_id)
        self.assertEqual(result[1].target_id, target_id)

    def test_success_when_get_relationships_by_type(self):
        """Given: 관계 타입이 제공될 때
        When: find_relationships를 호출하면
        Then: 해당 타입의 모든 관계가 반환된다
        """
        # Given
        relation_type = "CONNECTED_TO"
        mock_rows = [
            self._create_mock_row(5, 10, 20, relation_type),
            self._create_mock_row(6, 15, 25, relation_type),
        ]
        self.mock_cursor.fetchall.side_effect = [
            mock_rows,  # First call for relationships
            [self._create_mock_row_for_entity(10, "Source", "Person")],  # For source entity 10
            [self._create_mock_row_for_entity(20, "Target", "Location")],  # For target entity 20
            [self._create_mock_row_for_entity(15, "Source2", "Person")],  # For source entity 15
            [self._create_mock_row_for_entity(25, "Target2", "Location")],  # For target entity 25
        ]
        self.mock_cursor.fetchone.return_value = (2,)  # total_count

        # When
        result, total_count = self.manager.find_relationships(relation_type=relation_type)

        # Then
        self.assertEqual(len(result), 2)
        self.assertEqual(total_count, 2)
        self.assertEqual(result[0].relation_type, relation_type)
        self.assertEqual(result[1].relation_type, relation_type)

    def test_success_when_get_relationship_count(self):
        """Given: 데이터베이스에 관계가 있을 때
        When: get_relationship_count를 호출하면
        Then: 올바른 개수가 반환된다
        """
        # Given
        self.mock_cursor.fetchone.return_value = (42,)

        # When
        result = self.manager.get_relationship_count()

        # Then
        self.assertEqual(result, 42)
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("SELECT COUNT(*) FROM edges", call_args[0])

    def test_success_when_get_relationship_count_by_type(self):
        """Given: 특정 타입의 관계가 있을 때
        When: get_relationship_count_by_type을 호출하면
        Then: 해당 타입의 개수가 반환된다
        """
        # Given
        relation_type = "CONNECTED_TO"
        self.mock_cursor.fetchone.return_value = (15,)

        # When
        result = self.manager.get_relationship_count_by_type(relation_type)

        # Then
        self.assertEqual(result, 15)
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("SELECT COUNT(*) FROM edges WHERE relation_type = ?", call_args[0])
        self.assertEqual(call_args[1], (relation_type,))

    def test_success_when_bulk_create_relationships(self):
        """Given: 여러 관계 데이터가 제공될 때
        When: bulk_create_relationships를 호출하면
        Then: 모든 관계가 생성되고 ID 목록이 반환된다
        """
        # Given
        relationships_data = [
            (10, 20, "CONNECTED_TO", {"weight": 0.8}),
            (15, 25, "RELATES_TO", {"confidence": 0.9}),
            (30, 40, "INFLUENCES", {}),
        ]
        self.mock_cursor.lastrowid = 100  # 마지막 삽입된 ID

        # When
        result = self.manager.bulk_create_relationships(relationships_data)

        # Then
        self.assertEqual(len(result), 3)
        self.assertEqual(self.mock_cursor.execute.call_count, 6)

        # executemany가 아닌 개별 execute 호출 확인
        actual_calls = [call[0][0] for call in self.mock_cursor.execute.call_args_list]
        # There should be 3 INSERT INTO edges calls and 3 INSERT INTO vector_outbox calls
        insert_edge_count = sum(1 for call_str in actual_calls if "INSERT INTO edges" in call_str)
        insert_vector_outbox_count = sum(
            1 for call_str in actual_calls if "INSERT INTO vector_outbox" in call_str
        )

        self.assertEqual(insert_edge_count, 3)
        self.assertEqual(insert_vector_outbox_count, 3)

    def test_empty_list_when_bulk_create_relationships_empty(self):
        """Given: 빈 관계 데이터 목록이 제공될 때
        When: bulk_create_relationships를 호출하면
        Then: 빈 목록이 반환된다
        """
        # Given
        relationships_data = []

        # When
        result = self.manager.bulk_create_relationships(relationships_data)

        # Then
        self.assertEqual(result, [])
        self.mock_cursor.execute.assert_not_called()

    def _create_mock_row(
        self,
        rel_id: int,
        source_id: int,
        target_id: int,
        rel_type: str,
        properties: dict = None,
        source_name: str = None,
        source_type: str = None,
        target_name: str = None,
        target_type: str = None,
    ):
        """테스트용 Mock Row 생성 헬퍼 메서드."""
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
        if source_name is not None:
            data["source_name"] = source_name
        if source_type is not None:
            data["source_type"] = source_type
        if target_name is not None:
            data["target_name"] = target_name
        if target_type is not None:
            data["target_type"] = target_type

        mock_row.__getitem__.side_effect = data.get
        return mock_row

    def _create_mock_row_for_entity(
        self,
        entity_id: int,
        entity_name: str,
        entity_type: str,
    ):
        """테스트용 Mock Entity Row 생성 헬퍼 메서드."""
        mock_row = MagicMock(spec=sqlite3.Row)
        data = {
            "id": entity_id,
            "uuid": f"entity-{entity_id}",
            "name": entity_name,
            "type": entity_type,
            "properties": "{}",
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }

        def getitem_side_effect(key):
            return data.get(key)

        mock_row.__getitem__.side_effect = getitem_side_effect
        return mock_row


if __name__ == "__main__":
    unittest.main()
