"""
NodeId 값 객체 단위 테스트.
"""

import unittest
import uuid

from src.domain.value_objects.node_id import NodeId


class TestNodeId(unittest.TestCase):
    """NodeId 값 객체 테스트."""

    def test_create_node_id_with_valid_string(self):
        """유효한 문자열 값으로 NodeId 생성 테스트."""
        value = "test-node-id"
        node_id = NodeId(value)
        self.assertEqual(node_id.value, value)
        self.assertEqual(str(node_id), value)

    def test_create_node_id_with_empty_string_raises_error(self):
        """빈 문자열로 NodeId 생성 시 ValueError가 발생하는지 테스트."""
        with self.assertRaises(ValueError) as context:
            NodeId("")
        self.assertIn("NodeId value cannot be empty", str(context.exception))

    def test_create_node_id_with_non_string_raises_error(self):
        """문자열이 아닌 값으로 NodeId 생성 시 ValueError가 발생하는지 테스트."""
        with self.assertRaises(ValueError) as context:
            NodeId(123)
        self.assertIn("NodeId value must be a string", str(context.exception))

    def test_generate_creates_valid_uuid(self):
        """generate() 메서드가 유효한 UUID 기반 NodeId를 생성하는지 테스트."""
        node_id = NodeId.generate()
        self.assertIsInstance(node_id, NodeId)
        self.assertIsInstance(node_id.value, str)

        # 유효한 UUID 형식인지 검증
        try:
            uuid.UUID(node_id.value)
        except ValueError:
            self.fail("Generated NodeId should be a valid UUID")

    def test_generate_creates_unique_ids(self):
        """generate() 메서드가 고유한 ID를 생성하는지 테스트."""
        id1 = NodeId.generate()
        id2 = NodeId.generate()
        self.assertNotEqual(id1.value, id2.value)

    def test_equality(self):
        """NodeId 동등성 비교 테스트."""
        value = "test-id"
        id1 = NodeId(value)
        id2 = NodeId(value)
        id3 = NodeId("different-id")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_hash_consistency(self):
        """NodeId 해시 일관성 테스트."""
        value = "test-id"
        id1 = NodeId(value)
        id2 = NodeId(value)

        self.assertEqual(hash(id1), hash(id2))

        # 세트와 딕셔너리에서 사용 가능한지 확인
        id_set = {id1, id2}
        self.assertEqual(len(id_set), 1)

    def test_immutability(self):
        """NodeId 불변성 테스트."""
        node_id = NodeId("test-id")

        # value 값을 수정할 수 없어야 함
        with self.assertRaises(AttributeError):
            node_id.value = "new-value"


if __name__ == "__main__":
    unittest.main()
