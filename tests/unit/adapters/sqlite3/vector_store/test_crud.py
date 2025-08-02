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
        self.vector_store._connection = mock_connection

        vector = Vector([1.0, 2.0, 3.0])
        metadata = {"type": "test", "category": "example"}

        # When
        result = await self.vector_store.add_vector("vec_1", vector, metadata)

        # Then
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("INSERT OR REPLACE INTO test_vectors", call_args[0])
        self.assertEqual(call_args[1][0], "vec_1")  # vector_id
        # metadata는 JSON으로 변환되어야 함
        self.assertEqual(call_args[1][2], json.dumps(metadata))

    async def test_success_when_add_vector_without_metadata(self):
        """Given: 메타데이터 없이 벡터가 제공될 때
        When: add_vector를 호출하면
        Then: 벡터가 저장되고 메타데이터는 None이다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        vector = Vector([1.0, 2.0])

        # When
        result = await self.vector_store.add_vector("vec_2", vector)

        # Then
        self.assertTrue(result)
        call_args = mock_cursor.execute.call_args[0]
        self.assertIsNone(call_args[1][2])  # metadata should be None

    async def test_success_when_add_vectors_batch(self):
        """Given: 여러 벡터가 딕셔너리로 제공될 때
        When: add_vectors를 호출하면
        Then: 모든 벡터가 배치로 저장된다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        vectors = {"vec_1": Vector([1.0, 2.0]), "vec_2": Vector([3.0, 4.0])}
        metadata = {"vec_1": {"type": "A"}, "vec_2": {"type": "B"}}

        # When
        result = await self.vector_store.add_vectors(vectors, metadata)

        # Then
        self.assertTrue(result)
        mock_cursor.executemany.assert_called_once()
        call_args = mock_cursor.executemany.call_args[0]
        self.assertIn("INSERT OR REPLACE INTO test_vectors", call_args[0])
        self.assertEqual(len(call_args[1]), 2)  # 2 vectors

    async def test_success_when_get_vector(self):
        """Given: 저장된 벡터가 있을 때
        When: get_vector를 호출하면
        Then: Vector 객체를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        # Vector [1.0, 2.0, 3.0]을 bytes로 변환한 것을 모방
        vector_blob = self.vector_store._vector_to_blob(Vector([1.0, 2.0, 3.0]))
        mock_cursor.fetchone.return_value = (vector_blob,)
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

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
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.get_vector("nonexistent")

        # Then
        self.assertIsNone(result)

    async def test_success_when_get_vectors_multiple(self):
        """Given: 여러 벡터 ID가 제공될 때
        When: get_vectors를 호출하면
        Then: 딕셔너리로 결과를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        vector1_blob = self.vector_store._vector_to_blob(Vector([1.0, 2.0]))
        vector2_blob = self.vector_store._vector_to_blob(Vector([3.0, 4.0]))
        mock_cursor.fetchall.return_value = [("vec_1", vector1_blob), ("vec_2", vector2_blob)]
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.get_vectors(["vec_1", "vec_2", "vec_3"])

        # Then
        self.assertEqual(len(result), 3)
        self.assertEqual(result["vec_1"].values, [1.0, 2.0])
        self.assertEqual(result["vec_2"].values, [3.0, 4.0])
        self.assertIsNone(result["vec_3"])  # not found

    async def test_success_when_update_vector(self):
        """Given: 벡터 업데이트 요청이 있을 때
        When: update_vector를 호출하면
        Then: add_vector와 동일하게 동작한다
        """
        # Given
        vector = Vector([5.0, 6.0])
        metadata = {"updated": True}

        with patch.object(self.vector_store, "add_vector", return_value=True) as mock_add:
            # When
            result = await self.vector_store.update_vector("vec_1", vector, metadata)

            # Then
            self.assertTrue(result)
            mock_add.assert_called_once_with("vec_1", vector, metadata)

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
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.delete_vector("vec_1")

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
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.delete_vector("nonexistent")

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
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.delete_vectors(["vec_1", "vec_2", "vec_3"])

        # Then
        self.assertEqual(result, 2)

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
        self.vector_store._connection = mock_connection

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
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.vector_exists("nonexistent")

        # Then
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
