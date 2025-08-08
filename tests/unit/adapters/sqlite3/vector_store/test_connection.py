"""
SQLiteVectorStore 연결 및 초기화 기능 테스트.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.vector_store import SQLiteVectorStore

# 테스트 상수 정의
DEFAULT_TABLE_NAME = "test_vectors"
DEFAULT_DIMENSION = 128
CUSTOM_DIMENSION = 256
SMALL_DIMENSION = 64
DEFAULT_METRIC = "cosine"
EUCLIDEAN_METRIC = "euclidean"
OPTIMIZE_TRUE = True
OPTIMIZE_FALSE = False


class TestSQLiteVectorStoreConnection(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore 연결 및 초기화 테스트."""

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

    def test_success(self):
        """Given: VectorStore 초기화 파라미터가 제공될 때
        When: SQLiteVectorStore를 초기화하면
        Then: 설정이 올바르게 적용된다
        """
        # Given & When
        store = SQLiteVectorStore(
            db_path="/test/path.db", table_name="custom_vectors", optimize=False
        )

        # Then
        self.assertEqual(str(store.db_path), "/test/path.db")
        self.assertEqual(store.table_name, "custom_vectors")
        self.assertFalse(store.optimize)
        self.assertIsNone(store._connection)
        self.assertIsNone(store._dimension)
        self.assertEqual(store._metric, DEFAULT_METRIC)

    @patch("src.adapters.sqlite3.vector_store_base.DatabaseConnection")
    async def test_success_when_connect(self, mock_connection_class):
        """Given: 정상적인 데이터베이스 연결이 가능할 때
        When: connect를 호출하면
        Then: True를 반환하고 설정을 로드한다
        """
        # Given
        mock_connection_instance = Mock()
        mock_connection = Mock()
        mock_connection_instance.connect.return_value = mock_connection
        mock_connection_class.return_value = mock_connection_instance

        store = SQLiteVectorStore(str(self.db_path))

        # When
        result = await store.connect()

        # Then
        self.assertTrue(result)
        self.assertEqual(store._connection, mock_connection)

    @patch("src.adapters.sqlite3.vector_store_base.DatabaseConnection")
    async def test_false_when_connect_fails(self, mock_connection_class):
        """Given: 데이터베이스 연결에 실패할 때
        When: connect를 호출하면
        Then: False를 반환한다
        """
        # Given
        mock_connection_instance = Mock()
        mock_connection_instance.connect.side_effect = Exception("Connection failed")
        mock_connection_class.return_value = mock_connection_instance

        store = SQLiteVectorStore(str(self.db_path))

        # When
        result = await store.connect()

        # Then
        self.assertFalse(result)

    async def test_success_when_disconnect(self):
        """Given: 활성 연결이 있을 때
        When: disconnect를 호출하면
        Then: 연결이 해제되고 True를 반환한다
        """
        # Given
        mock_connection_manager = Mock()
        self.vector_store.writer._connection_manager = mock_connection_manager
        self.vector_store.writer._connection = Mock()

        # When
        result = await self.vector_store.disconnect()

        # Then
        self.assertTrue(result)
        self.assertIsNone(self.vector_store.writer._connection)
        mock_connection_manager.close.assert_called_once()

    async def test_true_when_is_connected(self):
        """Given: 정상적인 연결이 있을 때
        When: is_connected를 호출하면
        Then: True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_connection.execute.return_value.fetchone.return_value = (1,)
        self.vector_store.writer._connection = mock_connection

        # When
        result = await self.vector_store.is_connected()

        # Then
        self.assertTrue(result)

    async def test_false_when_not_connected(self):
        """Given: 연결이 없을 때
        When: is_connected를 호출하면
        Then: False를 반환한다
        """
        # Given
        self.vector_store._connection = None

        # When
        result = await self.vector_store.is_connected()

        # Then
        self.assertFalse(result)

    async def test_success_when_initialize_store(self):
        """Given: 정상적인 연결이 있을 때
        When: initialize_store를 호출하면
        Then: 벡터 스토어가 초기화되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        # Set connection on the writer component since initialize_store is delegated to it
        self.vector_store.writer._connection = mock_connection

        # When
        result = await self.vector_store.initialize_store(
            dimension=DEFAULT_DIMENSION, metric=EUCLIDEAN_METRIC, parameters={"index_type": "hnsw"}
        )

        # Then
        self.assertTrue(result)
        self.assertEqual(self.vector_store._dimension, DEFAULT_DIMENSION)
        self.assertEqual(self.vector_store._metric, EUCLIDEAN_METRIC)
        mock_cursor.execute.assert_called()
        mock_connection.commit.assert_called_once()

    async def test_false_when_initialize_store_auto_connect_fails(self):
        """Given: 연결이 없을 때
        When: initialize_store를 호출하면
        Then: 자동으로 연결을 시도한다
        """
        # Given
        self.vector_store.writer._connection = None
        with patch.object(self.vector_store.writer, "connect", return_value=False):
            # When
            result = await self.vector_store.initialize_store(dimension=SMALL_DIMENSION)

            # Then
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
