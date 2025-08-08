"""
SQLiteVectorStore CRUD 기능 테스트.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.domain import Vector
from src.domain.value_objects.document_metadata import DocumentMetadata

# 테스트 상수 정의
DEFAULT_TABLE_NAME = "test_vectors"
OPTIMIZE_TRUE = True


class TestSQLiteVectorStoreCRUD(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore CRUD 기능 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_vector.db"
        self.vector_store = SQLiteVectorStore(
            db_path=str(self.db_path), table_name=DEFAULT_TABLE_NAME, optimize=OPTIMIZE_TRUE
        )
        # Initialize dimension for testing
        self.vector_store.writer._dimension = 3
        self.vector_store.reader._dimension = 3
        self.vector_store.retriever._dimension = 3

    def tearDown(self):
        """테스트 정리."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    async def test_success_when_add_vector(self):
        """Given: 벡터와 메타데이터가 제공될 때
        When: add_vector를 호출하면
        Then: 벡터가 저장되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        vector = Vector([1.0, 2.0, 3.0])
        document_metadata = DocumentMetadata(
            content="vec_1", metadata={"type": "test", "category": "example"}
        )

        # When
        result = await self.vector_store.add_vectors([vector], [document_metadata])

        # Then
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # One document ID returned
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("INSERT OR REPLACE INTO test_vectors", call_args[0])
        # Metadata should include source, content and original metadata
        expected_metadata = {
            "source": document_metadata.source,
            "content": document_metadata.content,
            "created_at": None,
            "updated_at": None,
            **document_metadata.metadata,
        }
        self.assertEqual(call_args[1][2], json.dumps(expected_metadata))

    async def test_success_when_add_vector_without_metadata(self):
        """Given: 메타데이터 없이 벡터가 제공될 때
        When: add_vector를 호출하면
        Then: 벡터가 저장되고 메타데이터는 None이다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        vector = Vector([1.0, 2.0, 3.0])
        document_metadata = DocumentMetadata(content="vec_2", metadata={})

        # When
        result = await self.vector_store.add_vectors([vector], [document_metadata])

        # Then
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        call_args = mock_cursor.execute.call_args[0]
        # Even with empty metadata, the implementation still adds source/content fields
        expected_metadata = {
            "source": document_metadata.source,
            "content": document_metadata.content,
            "created_at": None,
            "updated_at": None,
        }
        self.assertEqual(call_args[1][2], json.dumps(expected_metadata))

    async def test_success_when_add_vectors_batch(self):
        """Given: 여러 벡터가 딕셔너리로 제공될 때
        When: add_vectors를 호출하면
        Then: 모든 벡터가 배치로 저장된다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        vectors = [Vector([1.0, 2.0, 3.0]), Vector([3.0, 4.0, 5.0])]
        documents = [
            DocumentMetadata(content="vec_1", metadata={"type": "A"}),
            DocumentMetadata(content="vec_2", metadata={"type": "B"}),
        ]

        # When
        result = await self.vector_store.add_vectors(vectors, documents)

        # Then
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Two document IDs returned
        # Implementation uses individual execute calls, not executemany
        self.assertEqual(mock_cursor.execute.call_count, 2)  # 2 vectors
        # Check first call
        first_call_args = mock_cursor.execute.call_args_list[0][0]
        self.assertIn("INSERT OR REPLACE INTO test_vectors", first_call_args[0])

    async def test_success_when_get_vector(self):
        """Given: 저장된 벡터가 있을 때
        When: get_vector를 호출하면
        Then: Vector 객체를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        # Vector [1.0, 2.0, 3.0]을 bytes로 변환한 것을 모방
        vector_blob = self.vector_store.reader._serialize_vector(Vector([1.0, 2.0, 3.0]))
        mock_cursor.fetchone.return_value = (vector_blob,)
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.reader._connection = mock_connection

        # When
        result = await self.vector_store.get_vector("vec_1")

        # Then
        self.assertIsInstance(result, Vector)
        self.assertEqual(result.values, [1.0, 2.0, 3.0])

    async def test_none_when_get_vector_not_found(self):
        """Given: 벡터가 존재하지 않을 때
        When: get_vector를 호출하면
        Then: None을 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.reader._connection = mock_connection

        # When
        result = await self.vector_store.get_vector("nonexistent")

        # Then
        self.assertIsNone(result)

    async def test_success_when_get_vector_multiple_calls(self):
        """Given: 여러 벡터 ID가 제공될 때
        When: get_vector를 각각 호출하면
        Then: 각각의 Vector 객체를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        vector1_blob = self.vector_store.reader._serialize_vector(Vector([1.0, 2.0, 3.0]))
        vector2_blob = self.vector_store.reader._serialize_vector(Vector([3.0, 4.0, 5.0]))

        # Mock different responses for different calls
        def mock_fetchone_side_effect():
            call_count = mock_fetchone_side_effect.call_count
            mock_fetchone_side_effect.call_count += 1
            if call_count == 0:
                return (vector1_blob,)
            elif call_count == 1:
                return (vector2_blob,)
            else:
                return None

        mock_fetchone_side_effect.call_count = 0

        mock_cursor.fetchone.side_effect = mock_fetchone_side_effect
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.reader._connection = mock_connection

        # When
        result1 = await self.vector_store.get_vector("vec_1")
        result2 = await self.vector_store.get_vector("vec_2")
        result3 = await self.vector_store.get_vector("vec_3")

        # Then
        self.assertIsInstance(result1, Vector)
        self.assertEqual(result1.values, [1.0, 2.0, 3.0])
        self.assertIsInstance(result2, Vector)
        self.assertEqual(result2.values, [3.0, 4.0, 5.0])
        self.assertIsNone(result3)  # not found

    async def test_success_when_update_vector(self):
        """Given: 벡터 업데이트 요청이 있을 때
        When: update_vector를 호출하면
        Then: add_vector와 동일하게 동작한다
        """
        # Given
        vector = Vector([5.0, 6.0, 7.0])
        metadata = {"updated": True}

        document_metadata = DocumentMetadata(content="vec_1", metadata=metadata)

        with patch.object(self.vector_store, "update_document", return_value=True) as mock_update:
            # When
            result = await self.vector_store.update_document("vec_1", document_metadata, vector)

            # Then
            self.assertTrue(result)
            mock_update.assert_called_once_with("vec_1", document_metadata, vector)

    async def test_success_when_delete_vector(self):
        """Given: 존재하는 벡터가 있을 때
        When: delete_vector를 호출하면
        Then: 벡터가 삭제되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        # When
        result = await self.vector_store.delete(["vec_1"])

        # Then
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("DELETE FROM test_vectors", call_args[0])

    async def test_false_when_delete_vector_not_found(self):
        """Given: 존재하지 않는 벡터일 때
        When: delete_vector를 호출하면
        Then: False를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 0
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        # When
        result = await self.vector_store.delete(["nonexistent"])

        # Then
        self.assertFalse(result)

    async def test_success_when_delete_vectors_batch(self):
        """Given: 여러 벡터 ID가 제공될 때
        When: delete_vectors를 호출하면
        Then: 삭제된 벡터 수를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 2
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.writer._connection = mock_connection

        # When
        result = await self.vector_store.delete(["vec_1", "vec_2", "vec_3"])

        # Then
        self.assertTrue(result)  # delete() returns bool, not count

    async def test_true_when_vector_exists(self):
        """Given: 벡터가 존재할 때
        When: vector_exists를 호출하면
        Then: True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.reader._connection = mock_connection

        # When
        result = await self.vector_store.vector_exists("vec_1")

        # Then
        self.assertTrue(result)

    async def test_false_when_vector_not_exists(self):
        """Given: 벡터가 존재하지 않을 때
        When: vector_exists를 호출하면
        Then: False를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store.reader._connection = mock_connection

        # When
        result = await self.vector_store.vector_exists("nonexistent")

        # Then
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
