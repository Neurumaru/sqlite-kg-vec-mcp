"""
SQLite 그래프 RelationshipManager CRUD 기능 테스트.
"""

import datetime
import json
import sqlite3
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.adapters.sqlite3.graph.relationships import (
    Relationship,
    RelationshipManager,
)


class TestRelationshipManagerCRUD(unittest.TestCase):
    """RelationshipManager CRUD 기능 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.manager = RelationshipManager(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결이 제공될 때
        When: RelationshipManager를 초기화하면
        Then: 연결이 올바르게 설정된다
        """
        # Given & When
        manager = RelationshipManager(self.mock_connection)

        # Then
        self.assertEqual(manager.connection, self.mock_connection)
        self.assertIsNotNone(manager.unit_of_work)

    def test_success_when_create_relationship(self):
        """Given: 유효한 관계 데이터가 제공될 때
        When: create_relationship를 호출하면
        Then: 관계가 생성되고 ID가 반환된다
        """
        # Given
        self.mock_cursor.lastrowid = 123
        source_id = 10
        target_id = 20
        relation_type = "CONNECTED_TO"
        properties = {"weight": 0.8}

        # Mock fetchone for entity existence check AND created relationship fetch
        mock_created_rel_row = MagicMock(spec=sqlite3.Row)
        row_data = {
            "id": self.mock_cursor.lastrowid,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "properties": json.dumps(properties),
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_created_rel_row.__getitem__.side_effect = row_data.get
        self.mock_cursor.fetchone.side_effect = [(2,), mock_created_rel_row]

        # When
        result = self.manager.create_relationship(source_id, target_id, relation_type, properties)

        # Then
        self.assertEqual(result.id, 123)
        self.assertEqual(result.source_id, source_id)
        self.assertEqual(result.target_id, target_id)
        self.assertEqual(result.relation_type, relation_type)
        self.assertEqual(result.properties, properties)

        # Verify execute calls
        self.assertEqual(
            self.mock_cursor.execute.call_count, 4
        )  # Count entities call + Insert relationship call + Vector operation + Select new relationship

    def test_success_when_create_relationship_without_properties(self):
        """Given: 프로퍼티 없이 관계 데이터가 제공될 때
        When: create_relationship를 호출하면
        Then: 빈 JSON으로 관계가 생성된다
        """
        # Given
        self.mock_cursor.lastrowid = 456
        source_id = 15
        target_id = 25
        relation_type = "RELATES_TO"

        # Mock fetchone for entity existence check AND created relationship fetch
        mock_created_rel_row = MagicMock(spec=sqlite3.Row)
        row_data = {
            "id": self.mock_cursor.lastrowid,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "properties": "{}",
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_created_rel_row.__getitem__.side_effect = row_data.get
        self.mock_cursor.fetchone.side_effect = [(2,), mock_created_rel_row]

        # When
        result = self.manager.create_relationship(source_id, target_id, relation_type)

        # Then
        self.assertEqual(result.id, 456)
        # Verify execute calls
        self.assertEqual(self.mock_cursor.execute.call_count, 4)
        insert_call = self.mock_cursor.execute.call_args_list[1]
        self.assertEqual(insert_call[0][1][3], "{}")  # Empty JSON

    def test_runtime_error_when_create_relationship_fails(self):
        """Given: 관계 생성이 실패할 때
        When: create_relationship를 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        self.mock_cursor.lastrowid = None  # Simulate insert failure
        source_id = 10
        target_id = 20
        relation_type = "INVALID"
        self.mock_cursor.fetchone.side_effect = [
            (2,),
            None,
        ]  # First fetch for count, second for created relationship (fails)

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            self.manager.create_relationship(source_id, target_id, relation_type)

        self.assertIn("엣지 삽입 실패", str(context.exception))

    def test_success_when_get_relationship_exists(self):
        """Given: 존재하는 관계 ID가 제공될 때
        When: get_relationship를 호출하면
        Then: 관계 객체가 반환된다
        """
        # Given
        relationship_id = 1
        mock_row = MagicMock()
        row_data = {
            "id": 1,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "CONNECTED_TO",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
        }
        mock_row.__getitem__.side_effect = row_data.get
        self.mock_cursor.fetchone.return_value = mock_row

        # When
        result = self.manager.get_relationship(relationship_id)

        # Then
        self.assertIsInstance(result, Relationship)
        self.assertEqual(result.id, relationship_id)
        self.mock_cursor.execute.assert_called_once()

    def test_none_when_get_relationship_not_exists(self):
        """Given: 존재하지 않는 관계 ID가 제공될 때
        When: get_relationship를 호출하면
        Then: None이 반환된다
        """
        # Given
        relationship_id = 999
        self.mock_cursor.fetchone.return_value = None

        # When
        result = self.manager.get_relationship(relationship_id)

        # Then
        self.assertIsNone(result)

    def test_success_when_update_relationship(self):
        """Given: 유효한 업데이트 데이터가 제공될 때
        When: update_relationship를 호출하면
        Then: 관계가 업데이트되고 True가 반환된다
        """
        # Given
        relationship_id = 1
        new_properties = {"weight": 0.9, "confidence": 0.95}
        self.mock_cursor.rowcount = 1

        # Mock existing relationship for get_relationship call
        mock_existing_rel_row = MagicMock(spec=sqlite3.Row)
        existing_row_data = {
            "id": relationship_id,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "OLD_TYPE",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_existing_rel_row.__getitem__.side_effect = existing_row_data.get

        # Mock updated relationship
        mock_updated_rel_row = MagicMock(spec=sqlite3.Row)
        updated_row_data = {
            "id": relationship_id,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "OLD_TYPE",
            "properties": json.dumps(new_properties),
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        mock_updated_rel_row.__getitem__.side_effect = updated_row_data.get

        # Mock transaction context
        mock_transaction_conn = Mock()
        mock_transaction_cursor = MagicMock(spec=sqlite3.Cursor)
        mock_transaction_cursor.rowcount = 1
        mock_transaction_cursor.fetchone.return_value = mock_updated_rel_row
        mock_transaction_conn.cursor.return_value = mock_transaction_cursor

        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        mock_unit_of_work.register_vector_operation.return_value = 1
        self.manager.unit_of_work = mock_unit_of_work

        with patch.object(
            self.manager,
            "get_relationship",
            return_value=Relationship.from_row(mock_existing_rel_row),
        ):
            # When
            result = self.manager.update_relationship(relationship_id, properties=new_properties)

            # Then
            self.assertTrue(result)
            self.assertEqual(mock_transaction_cursor.execute.call_count, 2)  # UPDATE + SELECT

    def test_false_when_update_relationship_not_found(self):
        """Given: 존재하지 않는 관계 ID가 제공될 때
        When: update_relationship를 호출하면
        Then: False가 반환된다
        """
        # Given
        relationship_id = 999
        self.mock_cursor.rowcount = 0

        with patch.object(self.manager, "get_relationship", return_value=None):
            # When
            result = self.manager.update_relationship(relationship_id, properties={"new": "value"})

            # Then
            self.assertFalse(result)

    def test_success_when_delete_relationship(self):
        """Given: 존재하는 관계 ID가 제공될 때
        When: delete_relationship를 호출하면
        Then: 관계가 삭제되고 True가 반환된다
        """
        # Given
        relationship_id = 1
        self.mock_cursor.rowcount = 1

        # When
        result = self.manager.delete_relationship(relationship_id)

        # Then
        self.assertTrue(result)
        self.assertEqual(self.mock_cursor.execute.call_count, 2)

    def test_false_when_delete_relationship_not_found(self):
        """Given: 존재하지 않는 관계 ID가 제공될 때
        When: delete_relationship를 호출하면
        Then: False가 반환된다
        """
        # Given
        relationship_id = 999
        self.mock_cursor.rowcount = 0

        # When
        result = self.manager.delete_relationship(relationship_id)

        # Then
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
