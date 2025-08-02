"""
SQLite 그래프 Relationship 데이터클래스 테스트.
"""

import datetime
import sqlite3
import unittest
from unittest.mock import MagicMock

from src.adapters.sqlite3.graph.entities import Entity
from src.adapters.sqlite3.graph.relationships import Relationship


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
        row_data = {
            "id": 2,
            "source_id": 15,
            "target_id": 25,
            "relation_type": "RELATES_TO",
            "properties": properties_dict,
            "created_at": datetime.datetime(2023, 12, 25, 11, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 11, 0, 0),
        }
        mock_row.__getitem__.side_effect = row_data.get

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
        row_data = {
            "id": 3,
            "source_id": 30,
            "target_id": 40,
            "relation_type": "CONTAINS",
            "properties": None,
            "created_at": datetime.datetime(2023, 12, 25, 12, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 12, 0, 0),
        }
        mock_row.__getitem__.side_effect = row_data.get

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
        row_data = {
            "id": 4,
            "source_id": 50,
            "target_id": 60,
            "relation_type": "INFLUENCES",
            "properties": "{}",
            "created_at": datetime.datetime(2023, 12, 25, 13, 0, 0),
            "updated_at": datetime.datetime(2023, 12, 25, 13, 0, 0),
        }
        mock_row.__getitem__.side_effect = row_data.get

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


if __name__ == "__main__":
    unittest.main()
