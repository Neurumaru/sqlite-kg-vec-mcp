"""
SQLite 그래프 관계(Relationship) 관리 모듈의 단위 테스트.
"""

import datetime
import json
import sqlite3
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.adapters.sqlite3.graph.entities import Entity
from src.adapters.sqlite3.graph.relationships import (
    Relationship,
    RelationshipManager,
)


class TestRelationship(unittest.TestCase):
    """Relationship 데이터클래스 테스트."""

    def test_from_row_with_json_properties(self):
        """Given: JSON 문자열 프로퍼티가 있는 데이터베이스 행이 있을 때
        When: from_row를 호출하면
        Then: 프로퍼티가 올바르게 파싱된 Relationship 객체가 생성된다
        """
        # Given
        mock_row = MagicMock(spec=sqlite3.Row)
        row_data = {
            "id": 1,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "CONNECTED_TO",
            "properties": '{"weight": 0.8, "confidence": 0.95}',
            "created_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
        }
        mock_row.__getitem__.side_effect = row_data.get

        # When
        relationship = Relationship.from_row(mock_row)

        # Then
        self.assertEqual(relationship.id, 1)
        self.assertEqual(relationship.source_id, 10)
        self.assertEqual(relationship.target_id, 20)
        self.assertEqual(relationship.relation_type, "CONNECTED_TO")
        self.assertEqual(relationship.properties["weight"], 0.8)
        self.assertEqual(relationship.properties["confidence"], 0.95)
        self.assertEqual(relationship.created_at, datetime.datetime(2023, 12, 25, 10, 0, 0))
        self.assertEqual(relationship.updated_at, datetime.datetime(2023, 12, 25, 10, 0, 0))
        self.assertIsNone(relationship.source)
        self.assertIsNone(relationship.target)

    def test_from_row_with_dict_properties(self):
        """Given: 이미 딕셔너리 형태의 프로퍼티가 있는 행이 있을 때
        When: from_row를 호출하면
        Then: 프로퍼티가 그대로 유지된 Relationship 객체가 생성된다
        """
        # Given
        properties_dict = {"weight": 0.7, "type": "semantic"}
        mock_row = MagicMock(spec=sqlite3.Row)
        mock_row.__getitem__.side_effect = lambda key: {
            "id": 2,
            "source_id": 15,
            "target_id": 25,
            "relation_type": "RELATES_TO",
            "properties": properties_dict,
            "created_at": datetime.datetime(2023, 12, 25, 11, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 11, 0, 0),
        }.get(key)

        # When
        relationship = Relationship.from_row(mock_row)

        # Then
        self.assertEqual(relationship.properties, properties_dict)

    def test_from_row_with_null_properties(self):
        """Given: NULL 프로퍼티가 있는 행이 있을 때
        When: from_row를 호출하면
        Then: 빈 딕셔너리 프로퍼티를 가진 Relationship 객체가 생성된다
        """
        # Given
        mock_row = MagicMock(spec=sqlite3.Row)
        mock_row.__getitem__.side_effect = lambda key: {
            "id": 3,
            "source_id": 30,
            "target_id": 40,
            "relation_type": "CONTAINS",
            "properties": None,
            "created_at": datetime.datetime(2023, 12, 25, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 12, 0, 0),
        }.get(key)

        # When
        relationship = Relationship.from_row(mock_row)

        # Then
        self.assertEqual(relationship.properties, {})

    def test_from_row_with_entity_details(self):
        """Given: 소스와 타겟 엔티티 정보가 포함된 Relationship이 있을 때
        When: 엔티티 정보를 설정하면
        Then: 관련 엔티티들이 올바르게 연결된다
        """
        # Given
        mock_row = MagicMock(spec=sqlite3.Row)
        mock_row.__getitem__.side_effect = lambda key: {
            "id": 4,
            "source_id": 50,
            "target_id": 60,
            "relation_type": "INFLUENCES",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 12, 25, 13, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 13, 0, 0),
        }.get(key)

        source_entity = Entity(
            id=50,
            uuid="source-uuid",
            name="Source Entity",
            type="Person",
            properties={},
            created_at="2023-12-25T13:00:00",
            updated_at="2023-12-25T13:00:00",
        )

        target_entity = Entity(
            id=60,
            uuid="target-uuid",
            name="Target Entity",
            type="Organization",
            properties={},
            created_at="2023-12-25T13:00:00",
            updated_at="2023-12-25T13:00:00",
        )

        # When
        relationship = Relationship.from_row(mock_row)
        relationship.source = source_entity
        relationship.target = target_entity

        # Then
        self.assertEqual(relationship.source, source_entity)
        self.assertEqual(relationship.target, target_entity)
        self.assertEqual(relationship.source_id, source_entity.id)
        self.assertEqual(relationship.target_id, target_entity.id)


class TestRelationshipManager(unittest.TestCase):
    """RelationshipManager 클래스 테스트."""

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

    def test_create_relationship_success(self):
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
        mock_created_rel_row.__getitem__.side_effect = lambda key: {
            "id": self.mock_cursor.lastrowid,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "properties": json.dumps(properties),
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }.get(key)
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
        # Verify the first execute call (COUNT(*))
        self.assertIn(
            "SELECT COUNT(*) FROM entities", self.mock_cursor.execute.call_args_list[0][0][0]
        )
        self.assertEqual(self.mock_cursor.execute.call_args_list[0][0][1], (source_id, target_id))

        # Verify the second execute call (INSERT)
        insert_call = self.mock_cursor.execute.call_args_list[1]
        self.assertIn("INSERT INTO edges", insert_call[0][0])
        self.assertEqual(
            insert_call[0][1], (source_id, target_id, relation_type, json.dumps(properties))
        )
        # Verify the third execute call (Vector operation)
        vector_op_call = self.mock_cursor.execute.call_args_list[2]
        self.assertIn("INSERT INTO vector_outbox", vector_op_call[0][0])
        self.assertIn("insert", vector_op_call[0][1])
        self.assertIn(self.mock_cursor.lastrowid, vector_op_call[0][1])
        # Verify the fourth execute call (SELECT new relationship)
        select_new_rel_call = self.mock_cursor.execute.call_args_list[3]
        self.assertIn("SELECT * FROM edges WHERE id = ?", select_new_rel_call[0][0])
        self.assertEqual(select_new_rel_call[0][1], (self.mock_cursor.lastrowid,))

    def test_create_relationship_without_properties(self):
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
        mock_created_rel_row.__getitem__.side_effect = lambda key: {
            "id": self.mock_cursor.lastrowid,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "properties": "{}",
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }.get(key)
        self.mock_cursor.fetchone.side_effect = [(2,), mock_created_rel_row]

        # When
        result = self.manager.create_relationship(source_id, target_id, relation_type)

        # Then
        self.assertEqual(result.id, 456)
        # Verify execute calls
        self.assertEqual(self.mock_cursor.execute.call_count, 4)
        insert_call = self.mock_cursor.execute.call_args_list[1]
        self.assertEqual(insert_call[0][1][3], "{}")  # Empty JSON

        select_new_rel_call = self.mock_cursor.execute.call_args_list[2]
        self.assertIn("SELECT * FROM edges WHERE id = ?", select_new_rel_call[0][0])
        self.assertEqual(select_new_rel_call[0][1], (self.mock_cursor.lastrowid,))

    def test_create_relationship_failure(self):
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

        self.assertIn("Failed to insert edge", str(context.exception))
        self.assertEqual(
            self.mock_cursor.execute.call_count, 2
        )  # Count entities call + Insert relationship call (fails)

    def test_get_relationship_by_id_exists(self):
        """Given: 존재하는 관계 ID가 제공될 때
        When: get_relationship를 호출하면
        Then: 관계 객체가 반환된다
        """
        # Given
        relationship_id = 1
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: {
            "id": 1,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "CONNECTED_TO",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 10, 0, 0),
        }.get(key)
        self.mock_cursor.fetchone.return_value = mock_row

        # When
        result = self.manager.get_relationship(relationship_id)

        # Then
        self.assertIsInstance(result, Relationship)
        self.assertEqual(result.id, relationship_id)
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn("SELECT * FROM edges WHERE id = ?", call_args[0])
        self.assertEqual(call_args[1], (relationship_id,))

    def test_get_relationship_by_id_not_exists(self):
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

    def test_get_relationships_by_source_success(self):
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

    def test_get_relationships_by_target_success(self):
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

    def test_get_relationships_by_type_success(self):
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

    def test_update_relationship_success(self):
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
        mock_existing_rel_row.__getitem__.side_effect = lambda key: {
            "id": relationship_id,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "OLD_TYPE",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }.get(key)
        # Mock fetchone for the updated relationship fetch inside update_relationship
        mock_updated_rel_row = MagicMock(spec=sqlite3.Row)
        mock_updated_rel_row.__getitem__.side_effect = lambda key: {
            "id": relationship_id,
            "source_id": 10,
            "target_id": 20,
            "relation_type": "OLD_TYPE",  # Type is not updated by this method
            "properties": json.dumps(new_properties),  # Properties are updated
            "created_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }.get(key)
        # Side effect for get_relationship (patched) and then cursor.fetchone (inside update_relationship)

        # Mock the cursor within the transaction to return the updated row
        mock_transaction_conn = Mock()
        mock_transaction_cursor = MagicMock(spec=sqlite3.Cursor)
        mock_transaction_cursor.rowcount = 1
        mock_transaction_cursor.fetchone.return_value = (
            mock_updated_rel_row  # Ensure fetchone returns the mocked row
        )
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
            # The `update_relationship` calls `get_relationship` (patched) and then executes 2 queries:
            # 1. UPDATE statement
            # 2. SELECT statement to fetch the updated relationship
            self.assertEqual(mock_transaction_cursor.execute.call_count, 2)  # UPDATE + SELECT
            # Verify the update call
            update_call_args = mock_transaction_cursor.execute.call_args_list[0][0]
            self.assertIn("UPDATE edges SET properties = ? WHERE id = ?", update_call_args[0])
            self.assertEqual(update_call_args[1], (json.dumps(new_properties), relationship_id))
            # Verify the fetch updated relationship call
            fetch_call_args = mock_transaction_cursor.execute.call_args_list[1][0]
            self.assertIn("SELECT * FROM edges WHERE id = ?", fetch_call_args[0])

    def test_update_relationship_not_found(self):
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

    def test_delete_relationship_success(self):
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
        # Verify the delete call
        delete_call_args = self.mock_cursor.execute.call_args_list[1][0]
        self.assertIn("DELETE FROM edges WHERE id = ?", delete_call_args[0])
        self.assertEqual(delete_call_args[1], (relationship_id,))
        # Verify the vector_outbox insert call
        vector_outbox_call_args = self.mock_cursor.execute.call_args_list[0][0]
        self.assertIn("INSERT INTO vector_outbox", vector_outbox_call_args[0])
        self.assertIn("delete", vector_outbox_call_args[1])
        self.assertIn(relationship_id, vector_outbox_call_args[1])

    def test_delete_relationship_not_found(self):
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

    def test_get_relationship_count_success(self):
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

    def test_get_relationship_count_by_type_success(self):
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

    def test_bulk_create_relationships_success(self):
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
        # The order of calls can be non-deterministic due to multiprocessing, assert for content.
        actual_calls = [call[0][0] for call in self.mock_cursor.execute.call_args_list]
        # There should be 3 INSERT INTO edges calls and 3 INSERT INTO vector_outbox calls
        insert_edge_count = sum(1 for call_str in actual_calls if "INSERT INTO edges" in call_str)
        insert_vector_outbox_count = sum(
            1 for call_str in actual_calls if "INSERT INTO vector_outbox" in call_str
        )

        self.assertEqual(insert_edge_count, 3)
        self.assertEqual(insert_vector_outbox_count, 3)

    def test_bulk_create_relationships_empty_list(self):
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

        mock_row.__getitem__.side_effect = lambda key: data.get(key)
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


class TestRelationshipManagerIntegration(unittest.TestCase):
    """RelationshipManager 통합 테스트."""

    def test_relationship_lifecycle(self):
        """Given: RelationshipManager와 실제 SQLite 연결이 있을 때
        When: 관계의 전체 생명주기를 테스트하면
        Then: 생성, 조회, 업데이트, 삭제가 모두 정상 작동한다
        """
        # 이 테스트는 실제 SQLite 데이터베이스를 사용하는 통합 테스트로
        # 단위 테스트 범위를 벗어나므로 스킵합니다.
        # 실제 구현에서는 임시 데이터베이스를 사용한 통합 테스트를 별도로 작성해야 합니다.
        self.skipTest("Integration test - requires actual database")


if __name__ == "__main__":
    unittest.main()
