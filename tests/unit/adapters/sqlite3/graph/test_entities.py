"""
Entity 및 EntityManager 단위 테스트.
"""

import sqlite3
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.adapters.sqlite3.graph.entities import Entity, EntityManager


class TestEntity(unittest.TestCase):
    """Entity 클래스 테스트."""

    def test_init(self):
        """Given: Entity 생성 파라미터가 주어질 때
        When: Entity를 생성하면
        Then: 모든 속성이 올바르게 설정된다
        """
        # Given
        entity_id = 1
        uuid_val = "uuid-123"
        name = "Test Entity"
        entity_type = "Person"
        properties = {"age": 30, "city": "Seoul"}
        created_at = "2024-01-01T00:00:00"
        updated_at = "2024-01-02T00:00:00"

        # When
        entity = Entity(
            id=entity_id,
            uuid=uuid_val,
            name=name,
            type=entity_type,
            properties=properties,
            created_at=created_at,
            updated_at=updated_at,
        )

        # Then
        self.assertEqual(entity.id, entity_id)
        self.assertEqual(entity.uuid, uuid_val)
        self.assertEqual(entity.name, name)
        self.assertEqual(entity.type, entity_type)
        self.assertEqual(entity.properties, properties)
        self.assertEqual(entity.created_at, created_at)
        self.assertEqual(entity.updated_at, updated_at)

    def test_from_row_with_json_string_properties(self):
        """Given: properties가 JSON 문자열인 데이터베이스 행이 있을 때
        When: from_row를 호출하면
        Then: JSON이 파싱되어 딕셔너리로 변환된다
        """
        # Given
        row = {
            "id": 1,
            "uuid": "uuid-123",
            "name": "Test Entity",
            "type": "Person",
            "properties": '{"age": 30, "city": "Seoul"}',
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = row.get

        # When
        entity = Entity.from_row(mock_row)

        # Then
        self.assertEqual(entity.properties, {"age": 30, "city": "Seoul"})

    def test_from_row_with_dict_properties(self):
        """Given: properties가 이미 딕셔너리인 데이터베이스 행이 있을 때
        When: from_row를 호출하면
        Then: 딕셔너리가 그대로 사용된다
        """
        # Given
        properties_dict = {"age": 25, "occupation": "Engineer"}
        row = {
            "id": 2,
            "uuid": "uuid-456",
            "name": "Another Entity",
            "type": "Professional",
            "properties": properties_dict,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = row.get

        # When
        entity = Entity.from_row(mock_row)

        # Then
        self.assertEqual(entity.properties, properties_dict)

    def test_from_row_with_null_properties(self):
        """Given: properties가 None인 데이터베이스 행이 있을 때
        When: from_row를 호출하면
        Then: 빈 딕셔너리가 설정된다
        """
        # Given
        row = {
            "id": 3,
            "uuid": "uuid-789",
            "name": "Entity Without Props",
            "type": "Empty",
            "properties": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = row.get

        # When
        entity = Entity.from_row(mock_row)

        # Then
        self.assertEqual(entity.properties, {})


class TestEntityManager(unittest.TestCase):
    """EntityManager 클래스 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.entity_manager = EntityManager(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결이 제공될 때
        When: EntityManager를 초기화하면
        Then: 연결과 UnitOfWork가 설정된다
        """
        # Given & When
        manager = EntityManager(self.mock_connection)

        # Then
        self.assertEqual(manager.connection, self.mock_connection)
        self.assertIsNotNone(manager.unit_of_work)

    @patch("src.adapters.sqlite3.graph.entities.uuid.uuid4")
    def test_create_entity_success(self, mock_uuid):
        """Given: 엔티티 생성 정보가 제공될 때
        When: create_entity를 호출하면
        Then: 새 엔티티가 생성되고 반환된다
        """
        # Given
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="generated-uuid")

        mock_cursor = Mock()
        mock_cursor.lastrowid = 123
        mock_cursor.fetchone.return_value = {
            "id": 123,
            "uuid": "generated-uuid",
            "name": "Test Entity",
            "type": "Person",
            "properties": '{"key": "value"}',
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        mock_transaction_conn = Mock()
        mock_transaction_conn.cursor.return_value = mock_cursor

        # UnitOfWork.begin() 컨텍스트 매니저 모킹
        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        mock_unit_of_work.register_vector_operation.return_value = 1
        self.entity_manager.unit_of_work = mock_unit_of_work

        # When
        result = self.entity_manager.create_entity(
            entity_type="Person", name="Test Entity", properties={"key": "value"}, custom_uuid=None
        )

        # Then
        self.assertIsInstance(result, Entity)
        self.assertEqual(result.id, 123)
        self.assertEqual(result.name, "Test Entity")
        self.assertEqual(result.type, "Person")
        mock_cursor.execute.assert_called()
        mock_unit_of_work.register_vector_operation.assert_called_once_with(
            entity_type="node", entity_id=123, operation_type="insert"
        )

    def test_create_entity_with_custom_uuid(self):
        """Given: 사용자 정의 UUID가 제공될 때
        When: create_entity를 호출하면
        Then: 제공된 UUID가 사용된다
        """
        # Given
        custom_uuid = "custom-uuid-123"

        mock_cursor = Mock()
        mock_cursor.lastrowid = 456
        mock_cursor.fetchone.return_value = {
            "id": 456,
            "uuid": custom_uuid,
            "name": "Custom UUID Entity",
            "type": "Test",
            "properties": "{}",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        mock_transaction_conn = Mock()
        mock_transaction_conn.cursor.return_value = mock_cursor

        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        mock_unit_of_work.register_vector_operation.return_value = 1
        self.entity_manager.unit_of_work = mock_unit_of_work

        # When
        result = self.entity_manager.create_entity(
            entity_type="Test", name="Custom UUID Entity", custom_uuid=custom_uuid
        )

        # Then
        self.assertEqual(result.uuid, custom_uuid)
        # UUID가 INSERT 쿼리에 사용되었는지 확인
        insert_call = mock_cursor.execute.call_args_list[0]
        self.assertEqual(insert_call[0][1][0], custom_uuid)  # 첫 번째 파라미터가 UUID

    def test_create_entity_lastrowid_none(self):
        """Given: lastrowid가 None일 때
        When: create_entity를 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = None

        mock_transaction_conn = Mock()
        mock_transaction_conn.cursor.return_value = mock_cursor

        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        self.entity_manager.unit_of_work = mock_unit_of_work

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            self.entity_manager.create_entity(entity_type="Test", name="Test")

        self.assertIn("Failed to insert entity", str(context.exception))

    def test_get_entity_success(self):
        """Given: 존재하는 엔티티 ID가 제공될 때
        When: get_entity를 호출하면
        Then: Entity 객체를 반환한다
        """
        # Given
        entity_id = 1
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {
            "id": entity_id,
            "uuid": "uuid-123",
            "name": "Found Entity",
            "type": "Person",
            "properties": '{"found": true}',
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.entity_manager.get_entity(entity_id)

        # Then
        self.assertIsInstance(result, Entity)
        self.assertEqual(result.id, entity_id)
        self.assertEqual(result.name, "Found Entity")
        mock_cursor.execute.assert_called_with("SELECT * FROM entities WHERE id = ?", (entity_id,))

    def test_get_entity_not_found(self):
        """Given: 존재하지 않는 엔티티 ID가 제공될 때
        When: get_entity를 호출하면
        Then: None을 반환한다
        """
        # Given
        entity_id = 999
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.entity_manager.get_entity(entity_id)

        # Then
        self.assertIsNone(result)

    def test_get_entity_by_uuid_success(self):
        """Given: 존재하는 UUID가 제공될 때
        When: get_entity_by_uuid를 호출하면
        Then: Entity 객체를 반환한다
        """
        # Given
        entity_uuid = "uuid-456"
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {
            "id": 2,
            "uuid": entity_uuid,
            "name": "UUID Found Entity",
            "type": "Organization",
            "properties": "{}",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.entity_manager.get_entity_by_uuid(entity_uuid)

        # Then
        self.assertIsInstance(result, Entity)
        self.assertEqual(result.uuid, entity_uuid)
        mock_cursor.execute.assert_called_with(
            "SELECT * FROM entities WHERE uuid = ?", (entity_uuid,)
        )

    def test_get_entity_by_uuid_not_found(self):
        """Given: 존재하지 않는 UUID가 제공될 때
        When: get_entity_by_uuid를 호출하면
        Then: None을 반환한다
        """
        # Given
        entity_uuid = "nonexistent-uuid"
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.entity_manager.get_entity_by_uuid(entity_uuid)

        # Then
        self.assertIsNone(result)

    def test_update_entity_success(self):
        """Given: 존재하는 엔티티와 업데이트 정보가 있을 때
        When: update_entity를 호출하면
        Then: 엔티티가 업데이트되고 반환된다
        """
        # Given
        entity_id = 1
        current_entity = Entity(
            id=entity_id,
            uuid="uuid-123",
            name="Old Name",
            type="Person",
            properties={"old": "value"},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )

        # get_entity 모킹
        with patch.object(self.entity_manager, "get_entity", return_value=current_entity):
            mock_cursor = Mock()
            mock_cursor.rowcount = 1
            mock_cursor.fetchone.return_value = {
                "id": entity_id,
                "uuid": "uuid-123",
                "name": "New Name",
                "type": "Person",
                "properties": '{"old": "value", "new": "data"}',
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
            }

            mock_transaction_conn = Mock()
            mock_transaction_conn.cursor.return_value = mock_cursor

            mock_unit_of_work = Mock()
            mock_transaction_context = MagicMock()
            mock_transaction_context.__enter__.return_value = mock_transaction_conn
            mock_transaction_context.__exit__.return_value = None
            mock_unit_of_work.begin.return_value = mock_transaction_context
            mock_unit_of_work.register_vector_operation.return_value = 1
            self.entity_manager.unit_of_work = mock_unit_of_work

            # When
            result = self.entity_manager.update_entity(
                entity_id=entity_id, name="New Name", properties={"new": "data"}
            )

            # Then
            self.assertIsInstance(result, Entity)
            self.assertEqual(result.name, "New Name")
            self.assertEqual(result.properties, {"old": "value", "new": "data"})
            mock_unit_of_work.register_vector_operation.assert_called_once_with(
                entity_type="node", entity_id=entity_id, operation_type="update"
            )

    def test_update_entity_not_found(self):
        """Given: 존재하지 않는 엔티티 ID가 제공될 때
        When: update_entity를 호출하면
        Then: None을 반환한다
        """
        # Given
        entity_id = 999

        with patch.object(self.entity_manager, "get_entity", return_value=None):
            # When
            result = self.entity_manager.update_entity(entity_id, name="New Name")

            # Then
            self.assertIsNone(result)

    def test_update_entity_no_changes(self):
        """Given: 변경사항이 없을 때
        When: update_entity를 호출하면
        Then: 현재 엔티티를 그대로 반환한다
        """
        # Given
        entity_id = 1
        current_entity = Entity(
            id=entity_id,
            uuid="uuid-123",
            name="Name",
            type="Person",
            properties={"key": "value"},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )

        with patch.object(self.entity_manager, "get_entity", return_value=current_entity):
            # When (name과 properties를 None으로 전달)
            result = self.entity_manager.update_entity(entity_id, name=None, properties=None)

            # Then
            self.assertEqual(result, current_entity)

    def test_delete_entity_success(self):
        """Given: 존재하는 엔티티 ID가 제공될 때
        When: delete_entity를 호출하면
        Then: 엔티티가 삭제되고 True를 반환한다
        """
        # Given
        entity_id = 1

        mock_cursor = Mock()
        mock_cursor.rowcount = 1

        mock_transaction_conn = Mock()
        mock_transaction_conn.cursor.return_value = mock_cursor

        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        mock_unit_of_work.register_vector_operation.return_value = 1
        self.entity_manager.unit_of_work = mock_unit_of_work

        # When
        result = self.entity_manager.delete_entity(entity_id)

        # Then
        self.assertTrue(result)
        mock_unit_of_work.register_vector_operation.assert_called_once_with(
            entity_type="node", entity_id=entity_id, operation_type="delete"
        )
        mock_cursor.execute.assert_called_with("DELETE FROM entities WHERE id = ?", (entity_id,))

    def test_delete_entity_not_found(self):
        """Given: 존재하지 않는 엔티티 ID가 제공될 때
        When: delete_entity를 호출하면
        Then: False를 반환한다
        """
        # Given
        entity_id = 999

        mock_cursor = Mock()
        mock_cursor.rowcount = 0

        mock_transaction_conn = Mock()
        mock_transaction_conn.cursor.return_value = mock_cursor

        mock_unit_of_work = Mock()
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_transaction_conn
        mock_transaction_context.__exit__.return_value = None
        mock_unit_of_work.begin.return_value = mock_transaction_context
        mock_unit_of_work.register_vector_operation.return_value = 1
        self.entity_manager.unit_of_work = mock_unit_of_work

        # When
        result = self.entity_manager.delete_entity(entity_id)

        # Then
        self.assertFalse(result)

    def test_find_entities_with_filters(self):
        """Given: 다양한 필터 조건이 제공될 때
        When: find_entities를 호출하면
        Then: 조건에 맞는 엔티티들과 총 개수를 반환한다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (2,)  # 총 개수
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "uuid": "uuid-1",
                "name": "Entity 1",
                "type": "Person",
                "properties": '{"age": 30}',
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            },
            {
                "id": 2,
                "uuid": "uuid-2",
                "name": "Entity 2",
                "type": "Person",
                "properties": '{"age": 25}',
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            },
        ]
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        entities, total_count = self.entity_manager.find_entities(
            entity_type="Person",
            name_pattern="Entity%",
            property_filters={"status": "active"},
            limit=10,
            offset=0,
        )

        # Then
        self.assertEqual(len(entities), 2)
        self.assertEqual(total_count, 2)
        self.assertIsInstance(entities[0], Entity)
        self.assertEqual(entities[0].type, "Person")

        # 쿼리가 올바르게 구성되었는지 확인
        calls = mock_cursor.execute.call_args_list
        self.assertEqual(len(calls), 2)  # count query + select query

        # count query 검증
        count_query = calls[0][0][0]
        self.assertIn("COUNT(DISTINCT id)", count_query)
        self.assertIn("type = ?", count_query)
        self.assertIn("name LIKE ?", count_query)
        self.assertIn("JSON_EXTRACT(properties, '$.status') = ?", count_query)

        # select query 검증
        select_query = calls[1][0][0]
        self.assertIn("SELECT DISTINCT * FROM entities", select_query)
        self.assertIn("ORDER BY id DESC", select_query)
        self.assertIn("LIMIT ? OFFSET ?", select_query)

    def test_find_entities_no_filters(self):
        """Given: 필터 조건이 없을 때
        When: find_entities를 호출하면
        Then: 모든 엔티티를 반환한다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "uuid": "uuid-1",
                "name": "Entity 1",
                "type": "Any",
                "properties": "{}",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
        ]
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        entities, total_count = self.entity_manager.find_entities()

        # Then
        self.assertEqual(len(entities), 1)
        self.assertEqual(total_count, 1)

        # WHERE 절이 없는 쿼리인지 확인
        calls = mock_cursor.execute.call_args_list
        count_query = calls[0][0][0]
        select_query = calls[1][0][0]
        self.assertNotIn("WHERE", count_query)
        self.assertNotIn("WHERE", select_query)


if __name__ == "__main__":
    unittest.main()
