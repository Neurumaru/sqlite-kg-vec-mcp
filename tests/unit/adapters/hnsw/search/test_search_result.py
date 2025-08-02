"""
HNSW 검색 어댑터의 SearchResult 관련 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

import unittest

from src.adapters.hnsw.search import SearchResult
from src.dto.node import NodeData, NodeType


class TestSearchResultInit(unittest.TestCase):
    """SearchResult.__init__ 메서드의 단위 테스트."""

    def test_success(self):
        """Given: SearchResult 매개변수들
        When: SearchResult를 초기화하면
        Then: 모든 속성이 설정되어야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        distance = 0.5
        entity = {"id": 123, "name": "test"}

        # When
        result = SearchResult(entity_type, entity_id, distance, entity)

        # Then
        self.assertEqual(result.entity_type, entity_type)
        self.assertEqual(result.entity_id, entity_id)
        self.assertEqual(result.distance, distance)
        self.assertEqual(result.entity, entity)


class TestSearchResultToDict(unittest.TestCase):
    """SearchResult.to_dict 메서드의 단위 테스트."""

    def test_success(self):
        """Given: 엔티티 정보 없는 SearchResult
        When: to_dict()를 호출하면
        Then: 기본 정보만 포함된 딕셔너리를 반환해야 한다
        """
        # Given
        result = SearchResult("node", 123, 0.3)

        # When
        result_dict = result.to_dict()

        # Then
        expected = {"entity_type": "node", "entity_id": 123, "distance": 0.3}
        self.assertEqual(result_dict, expected)

    def test_success_when_with_entity(self):
        """Given: 엔티티 정보 포함된 SearchResult
        When: to_dict()를 호출하면
        Then: 엔티티 정보도 포함된 딕셔너리를 반환해야 한다
        """
        # Given
        test_entity = NodeData(
            id="123", name="Test Entity", node_type=NodeType.PERSON, properties={"age": 30}
        )

        result = SearchResult("node", 123, 0.3, test_entity)

        # When
        result_dict = result.to_dict()

        # Then
        self.assertEqual(result_dict["entity_type"], "node")
        self.assertEqual(result_dict["entity_id"], 123)
        self.assertEqual(result_dict["distance"], 0.3)
        self.assertIn("entity", result_dict)
        self.assertEqual(result_dict["entity"]["id"], "123")
        self.assertEqual(result_dict["entity"]["name"], "Test Entity")


if __name__ == "__main__":
    unittest.main()
