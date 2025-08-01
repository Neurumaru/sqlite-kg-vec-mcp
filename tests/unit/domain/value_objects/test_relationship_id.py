"""
RelationshipId 값 객체 단위 테스트.
"""

import unittest
import uuid

from src.domain.value_objects.relationship_id import RelationshipId


class TestRelationshipId(unittest.TestCase):
    """RelationshipId 값 객체 테스트."""

    def test_create_relationship_id_success(self):
        """RelationshipId 생성 성공 테스트."""
        # When
        value = "test-relationship-id"
        relationship_id = RelationshipId(value)

        # Then
        self.assertEqual(relationship_id.value, value)
        self.assertEqual(str(relationship_id), value)

    def test_create_relationship_id_with_empty_string_error(self):
        """빈 문자열로 RelationshipId 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            RelationshipId("")
        self.assertIn("RelationshipId cannot be empty", str(context.exception))

    def test_create_relationship_id_with_non_string_error(self):
        """문자열이 아닌 값으로 RelationshipId 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            RelationshipId(123)
        self.assertIn("RelationshipId must be a string", str(context.exception))

    def test_generate_creates_valid_uuid(self):
        """generate 메서드가 유효한 UUID 기반 RelationshipId를 생성하는지 테스트."""
        # When
        relationship_id = RelationshipId.generate()

        # Then
        self.assertIsInstance(relationship_id, RelationshipId)
        self.assertIsInstance(relationship_id.value, str)

        # UUID 형식 검증
        try:
            uuid.UUID(relationship_id.value)
        except ValueError:
            self.fail("Generated RelationshipId should be a valid UUID")

    def test_generate_creates_unique_ids(self):
        """generate 메서드가 고유한 ID를 생성하는지 테스트."""
        # When
        id1 = RelationshipId.generate()
        id2 = RelationshipId.generate()

        # Then
        self.assertNotEqual(id1.value, id2.value)

    def test_from_string_success(self):
        """from_string 메서드 성공 테스트."""
        # When
        value = "test-relationship-id"
        relationship_id = RelationshipId.from_string(value)

        # Then
        self.assertIsInstance(relationship_id, RelationshipId)
        self.assertEqual(relationship_id.value, value)

    def test_equality(self):
        """RelationshipId 동등성 비교 테스트."""
        value = "test-id"
        id1 = RelationshipId(value)
        id2 = RelationshipId(value)
        id3 = RelationshipId("different-id")

        # Then
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_hash_consistency(self):
        """RelationshipId 해시 일관성 테스트."""
        value = "test-id"
        id1 = RelationshipId(value)
        id2 = RelationshipId(value)

        # Then
        self.assertEqual(hash(id1), hash(id2))

        # 세트와 딕셔너리에서 사용 가능한지 확인
        id_set = {id1, id2}
        self.assertEqual(len(id_set), 1)

    def test_immutability(self):
        """RelationshipId 불변성 테스트."""
        relationship_id = RelationshipId("test-id")

        # Then
        # value 값을 수정할 수 없어야 함
        with self.assertRaises(AttributeError):
            relationship_id.value = "new-value"

    def test_repr_representation(self):
        """repr 표현 테스트."""
        value = "test-relationship-id"
        relationship_id = RelationshipId(value)

        # When
        repr_str = repr(relationship_id)

        # Then
        self.assertIn("RelationshipId", repr_str)
        self.assertIn(value, repr_str)


if __name__ == "__main__":
    unittest.main()
