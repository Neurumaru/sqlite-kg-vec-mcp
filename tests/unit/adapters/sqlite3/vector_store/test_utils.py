"""
SQLiteVectorStore 유틸리티 및 관리 기능 테스트.
"""

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


class TestSQLiteVectorStoreUtils(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore 유틸리티 및 관리 기능 테스트."""

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

    async def test_success_when_get_store_info(self):
        """Given: 벡터 스토어가 초기화되어 있을 때
        When: get_store_info를 호출하면
        Then: 스토어 정보를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (100,)  # 벡터 개수
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection
        self.vector_store._dimension = 128
        self.vector_store._metric = "cosine"

        # When
        result = await self.vector_store.get_store_info()

        # Then
        self.assertEqual(result["table_name"], "test_vectors")
        self.assertEqual(result["dimension"], 128)
        self.assertEqual(result["metric"], "cosine")
        self.assertEqual(result["vector_count"], 100)

    async def test_success_when_get_vector_count(self):
        """Given: 벡터들이 저장되어 있을 때
        When: get_vector_count를 호출하면
        Then: 벡터 개수를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (42,)
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.get_vector_count()

        # Then
        self.assertEqual(result, 42)

    async def test_success_when_get_dimension(self):
        """Given: 차원이 설정되어 있을 때
        When: get_dimension을 호출하면
        Then: 차원 수를 반환한다
        """
        # Given
        self.vector_store._dimension = 256

        # When
        result = await self.vector_store.get_dimension()

        # Then
        self.assertEqual(result, 256)

    async def test_success_when_optimize_store(self):
        """Given: 정상적인 연결이 있을 때
        When: optimize_store를 호출하면
        Then: 최적화가 수행되고 결과를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.optimize_store()

        # Then
        self.assertEqual(result["status"], "optimized")
        self.assertIn("vacuum", result["operations"])
        self.assertIn("analyze", result["operations"])
        mock_cursor.execute.assert_any_call("VACUUM")
        mock_cursor.execute.assert_any_call("ANALYZE test_vectors")

    async def test_success_when_clear_store(self):
        """Given: 벡터들이 저장되어 있을 때
        When: clear_store를 호출하면
        Then: 모든 벡터가 삭제되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.clear_store()

        # Then
        self.assertTrue(result)
        mock_cursor.execute.assert_called_with("DELETE FROM test_vectors")

    async def test_healthy_when_health_check(self):
        """Given: 모든 상태가 정상일 때
        When: health_check를 호출하면
        Then: healthy 상태를 반환한다
        """
        # Given
        self.vector_store._dimension = 128  # config loaded

        with patch.object(self.vector_store, "is_connected", return_value=True):
            mock_connection = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = ("test_vectors",)
            mock_connection.cursor.return_value = mock_cursor
            self.vector_store._connection = mock_connection

            # When
            result = await self.vector_store.health_check()

            # Then
            self.assertEqual(result["status"], "healthy")
            self.assertTrue(result["connected"])
            self.assertTrue(result["table_exists"])
            self.assertTrue(result["config_loaded"])

    def test_success_when_vector_to_blob_conversion(self):
        """Given: Vector 객체가 있을 때
        When: _vector_to_blob을 호출하면
        Then: bytes로 변환된다
        """
        # Given
        vector = Vector([1.0, 2.0, 3.0])

        # When
        blob = self.vector_store._vector_to_blob(vector)

        # Then
        self.assertIsInstance(blob, bytes)
        self.assertGreater(len(blob), 0)

    def test_success_when_blob_to_vector_conversion(self):
        """Given: 벡터 blob이 있을 때
        When: _blob_to_vector를 호출하면
        Then: Vector 객체로 변환된다
        """
        # Given
        original_vector = Vector([1.0, 2.0, 3.0])
        blob = self.vector_store._vector_to_blob(original_vector)

        # When
        converted_vector = self.vector_store._blob_to_vector(blob)

        # Then
        self.assertIsInstance(converted_vector, Vector)
        self.assertEqual(converted_vector.values, [1.0, 2.0, 3.0])

    async def test_success_when_load_config(self):
        """Given: 설정이 저장되어 있을 때
        When: _load_config를 호출하면
        Then: 설정이 로드된다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (128, "euclidean", '{"param": "value"}')
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        await self.vector_store._load_config()

        # Then
        self.assertEqual(self.vector_store._dimension, 128)
        self.assertEqual(self.vector_store._metric, "euclidean")

    async def test_success_when_load_config_not_found(self):
        """Given: 설정이 저장되어 있지 않을 때
        When: _load_config를 호출하면
        Then: 예외 없이 처리된다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None  # 설정 없음
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When & Then (예외가 발생하지 않아야 함)
        await self.vector_store._load_config()


if __name__ == "__main__":
    unittest.main()
