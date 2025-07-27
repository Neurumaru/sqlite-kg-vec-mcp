"""
DocumentId 값 객체 단위 테스트.
"""

import unittest
import uuid

from src.domain.value_objects.document_id import DocumentId


class TestDocumentId(unittest.TestCase):
    """DocumentId 값 객체 테스트."""

    def test_create_document_id_success(self):
        """DocumentId 생성 성공 테스트."""
        # When
        value = "test-document-id"
        document_id = DocumentId(value)

        # Then
        self.assertEqual(document_id.value, value)
        self.assertEqual(str(document_id), value)

    def test_create_document_id_with_empty_string_error(self):
        """빈 문자열로 DocumentId 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            DocumentId("")
        self.assertIn("DocumentId cannot be empty", str(context.exception))

    def test_create_document_id_with_non_string_error(self):
        """문자열이 아닌 값으로 DocumentId 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            DocumentId(123)
        self.assertIn("DocumentId must be a string", str(context.exception))

    def test_generate_creates_valid_uuid(self):
        """generate 메서드가 유효한 UUID 기반 DocumentId를 생성하는지 테스트."""
        # When
        document_id = DocumentId.generate()

        # Then
        self.assertIsInstance(document_id, DocumentId)
        self.assertIsInstance(document_id.value, str)

        # UUID 형식 검증
        try:
            uuid.UUID(document_id.value)
        except ValueError:
            self.fail("Generated DocumentId should be a valid UUID")

    def test_generate_creates_unique_ids(self):
        """generate 메서드가 고유한 ID를 생성하는지 테스트."""
        # When
        id1 = DocumentId.generate()
        id2 = DocumentId.generate()

        # Then
        self.assertNotEqual(id1.value, id2.value)

    def test_from_string_success(self):
        """from_string 메서드 성공 테스트."""
        # When
        value = "test-document-id"
        document_id = DocumentId.from_string(value)

        # Then
        self.assertIsInstance(document_id, DocumentId)
        self.assertEqual(document_id.value, value)

    def test_equality(self):
        """DocumentId 동등성 비교 테스트."""
        value = "test-id"
        id1 = DocumentId(value)
        id2 = DocumentId(value)
        id3 = DocumentId("different-id")

        # Then
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_hash_consistency(self):
        """DocumentId 해시 일관성 테스트."""
        value = "test-id"
        id1 = DocumentId(value)
        id2 = DocumentId(value)

        # Then
        self.assertEqual(hash(id1), hash(id2))

        # 세트와 딕셔너리에서 사용 가능한지 확인
        id_set = {id1, id2}
        self.assertEqual(len(id_set), 1)

    def test_immutability(self):
        """DocumentId 불변성 테스트."""
        document_id = DocumentId("test-id")

        # Then
        # value 값을 수정할 수 없어야 함
        with self.assertRaises(AttributeError):
            document_id.value = "new-value"

    def test_repr_representation(self):
        """repr 표현 테스트."""
        value = "test-document-id"
        document_id = DocumentId(value)

        # When
        repr_str = repr(document_id)

        # Then
        self.assertIn("DocumentId", repr_str)
        self.assertIn(value, repr_str)


if __name__ == "__main__":
    unittest.main()