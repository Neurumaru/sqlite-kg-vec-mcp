"""
SQLiteVectorStore 단위 테스트.
"""

# pylint: disable=protected-access

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.domain import Vector

# 테스트 상수 정의
DEFAULT_TABLE_NAME = "test_vectors"
DEFAULT_DIMENSION = 128
CUSTOM_DIMENSION = 256
SMALL_DIMENSION = 64
DEFAULT_METRIC = "cosine"
EUCLIDEAN_METRIC = "euclidean"
OPTIMIZE_TRUE = True
OPTIMIZE_FALSE = False
EXPECTED_FLOAT_PRECISION = 6


class TestSQLiteVectorStore(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_vector.db"
        self.vector_store = SQLiteVectorStore(
            db_path=str(self.db_path), table_name=DEFAULT_TABLE_NAME, optimize=OPTIMIZE_TRUE
        )

    def tearDown(self):
        """테스트 정리."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
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

    @patch("src.adapters.sqlite3.vector_store.DatabaseConnection")
    async def test_connect_success(self, mock_connection_class):
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

    @patch("src.adapters.sqlite3.vector_store.DatabaseConnection")
    async def test_connect_failure(self, mock_connection_class):
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

    async def test_disconnect_success(self):
        """Given: 활성 연결이 있을 때
        When: disconnect를 호출하면
        Then: 연결이 해제되고 True를 반환한다
        """
        # Given
        mock_connection_manager = Mock()
        self.vector_store._connection_manager = mock_connection_manager
        self.vector_store._connection = Mock()

        # When
        result = await self.vector_store.disconnect()

        # Then
        self.assertTrue(result)
        self.assertIsNone(self.vector_store._connection)
        mock_connection_manager.close.assert_called_once()

    async def test_is_connected_true(self):
        """Given: 정상적인 연결이 있을 때
        When: is_connected를 호출하면
        Then: True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_connection.execute.return_value.fetchone.return_value = (1,)
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.is_connected()

        # Then
        self.assertTrue(result)

    async def test_is_connected_false_no_connection(self):
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

    async def test_initialize_store_success(self):
        """Given: 정상적인 연결이 있을 때
        When: initialize_store를 호출하면
        Then: 벡터 스토어가 초기화되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

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

    async def test_initialize_store_auto_connect(self):
        """Given: 연결이 없을 때
        When: initialize_store를 호출하면
        Then: 자동으로 연결을 시도한다
        """
        # Given
        self.vector_store._connection = None
        with patch.object(self.vector_store, "connect", return_value=False):
            # When
            result = await self.vector_store.initialize_store(dimension=SMALL_DIMENSION)

            # Then
            self.assertFalse(result)

    async def test_add_vector_success(self):
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

    async def test_add_vector_without_metadata(self):
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

    async def test_add_vectors_batch(self):
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

    async def test_get_vector_success(self):
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

    async def test_get_vector_not_found(self):
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

    async def test_get_vectors_multiple(self):
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

    async def test_update_vector(self):
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

    async def test_delete_vector_success(self):
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

    async def test_delete_vector_not_found(self):
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

    async def test_delete_vectors_batch(self):
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

    async def test_vector_exists_true(self):
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

    async def test_vector_exists_false(self):
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

    async def test_search_similar_basic(self):
        """Given: 쿼리 벡터와 저장된 벡터들이 있을 때
        When: search_similar을 호출하면
        Then: 유사도 순으로 결과를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        # 두 개의 벡터를 반환하도록 설정
        vector1_blob = self.vector_store._vector_to_blob(Vector([1.0, 0.0]))
        vector2_blob = self.vector_store._vector_to_blob(Vector([0.0, 1.0]))
        mock_cursor.fetchall.return_value = [("vec_1", vector1_blob), ("vec_2", vector2_blob)]
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        query_vector = Vector([1.0, 0.0])  # vec_1과 동일

        # When
        result = await self.vector_store.search_similar(query_vector, k=2)

        # Then
        self.assertEqual(len(result), 2)
        # vec_1이 더 유사해야 함 (동일한 벡터)
        self.assertEqual(result[0][0], "vec_1")
        self.assertGreater(result[0][1], result[1][1])  # 더 높은 유사도

    async def test_search_similar_with_filters(self):
        """Given: 필터 조건이 있을 때
        When: search_similar을 호출하면
        Then: 필터가 적용된 결과를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        vector_blob = self.vector_store._vector_to_blob(Vector([1.0, 0.0]))
        mock_cursor.fetchall.return_value = [("vec_1", vector_blob)]
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        query_vector = Vector([1.0, 0.0])
        filters = {"type": "test", "category": "A"}

        # When
        _ = await self.vector_store.search_similar(query_vector, k=5, filter_criteria=filters)

        # Then
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("WHERE", call_args[0])
        self.assertIn("json_extract(metadata, '$.type') = ?", call_args[0])

    async def test_search_similar_with_vectors(self):
        """Given: 벡터 자체가 필요할 때
        When: search_similar_with_vectors를 호출하면
        Then: 벡터와 함께 결과를 반환한다
        """
        # Given
        query_vector = Vector([1.0, 0.0])

        with (
            patch.object(
                self.vector_store, "search_similar", return_value=[("vec_1", 0.9), ("vec_2", 0.8)]
            ),
            patch.object(
                self.vector_store,
                "get_vectors",
                return_value={"vec_1": Vector([1.0, 0.0]), "vec_2": Vector([0.0, 1.0])},
            ),
        ):
            # When
            result = await self.vector_store.search_similar_with_vectors(query_vector, k=2)

            # Then
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0][0], "vec_1")
            self.assertEqual(result[0][1].values, [1.0, 0.0])
            self.assertEqual(result[0][2], 0.9)

    async def test_search_by_ids(self):
        """Given: 특정 ID 목록이 제공될 때
        When: search_by_ids를 호출하면
        Then: 해당 ID들 중에서만 검색한다
        """
        # Given
        query_vector = Vector([1.0, 0.0])
        candidate_ids = ["vec_1", "vec_2"]

        with patch.object(
            self.vector_store,
            "get_vectors",
            return_value={"vec_1": Vector([1.0, 0.0]), "vec_2": Vector([0.0, 1.0])},
        ) as mock_get:
            # When
            result = await self.vector_store.search_by_ids(query_vector, candidate_ids)

            # Then
            self.assertEqual(len(result), 2)
            mock_get.assert_called_once_with(candidate_ids)

    async def test_batch_search(self):
        """Given: 여러 쿼리 벡터가 제공될 때
        When: batch_search를 호출하면
        Then: 각 쿼리에 대한 결과 리스트를 반환한다
        """
        # Given
        query_vectors = [Vector([1.0, 0.0]), Vector([0.0, 1.0])]

        with patch.object(
            self.vector_store, "search_similar", side_effect=[[("vec_1", 0.9)], [("vec_2", 0.8)]]
        ) as mock_search:
            # When
            result = await self.vector_store.batch_search(query_vectors, k=1)

            # Then
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], [("vec_1", 0.9)])
            self.assertEqual(result[1], [("vec_2", 0.8)])
            self.assertEqual(mock_search.call_count, 2)

    async def test_get_metadata_success(self):
        """Given: 메타데이터가 있는 벡터가 존재할 때
        When: get_metadata를 호출하면
        Then: 메타데이터 딕셔너리를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        metadata = {"type": "test", "value": 123}
        mock_cursor.fetchone.return_value = (json.dumps(metadata),)
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.get_metadata("vec_1")

        # Then
        self.assertEqual(result, metadata)

    async def test_get_metadata_not_found(self):
        """Given: 벡터가 존재하지 않을 때
        When: get_metadata를 호출하면
        Then: None을 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        # When
        result = await self.vector_store.get_metadata("nonexistent")

        # Then
        self.assertIsNone(result)

    async def test_update_metadata_success(self):
        """Given: 벡터가 존재할 때
        When: update_metadata를 호출하면
        Then: 메타데이터가 업데이트되고 True를 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        new_metadata = {"updated": True, "version": 2}

        # When
        result = await self.vector_store.update_metadata("vec_1", new_metadata)

        # Then
        self.assertTrue(result)
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("UPDATE test_vectors", call_args[0])
        self.assertEqual(call_args[1][0], json.dumps(new_metadata))

    async def test_search_by_metadata(self):
        """Given: 메타데이터 필터가 제공될 때
        When: search_by_metadata를 호출하면
        Then: 조건에 맞는 벡터 ID들을 반환한다
        """
        # Given
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("vec_1",), ("vec_2",)]
        mock_connection.cursor.return_value = mock_cursor
        self.vector_store._connection = mock_connection

        filters = {"type": "test", "status": "active"}

        # When
        result = await self.vector_store.search_by_metadata(filters, limit=10)

        # Then
        self.assertEqual(result, ["vec_1", "vec_2"])
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn("json_extract(metadata, '$.type') = ?", call_args[0])
        self.assertIn("json_extract(metadata, '$.status') = ?", call_args[0])

    async def test_get_store_info(self):
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

    async def test_get_vector_count(self):
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

    async def test_get_dimension(self):
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

    async def test_optimize_store_success(self):
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

    async def test_clear_store_success(self):
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

    async def test_health_check_healthy(self):
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

    def test_vector_to_blob_conversion(self):
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

    def test_blob_to_vector_conversion(self):
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

    def test_calculate_similarity_identical_vectors(self):
        """Given: 동일한 두 벡터가 있을 때
        When: _calculate_similarity를 호출하면
        Then: 1.0의 유사도를 반환한다
        """
        # Given
        vector1 = Vector([1.0, 0.0, 0.0])
        vector2 = Vector([1.0, 0.0, 0.0])

        # When
        similarity = self.vector_store._calculate_similarity(vector1, vector2)

        # Then
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_calculate_similarity_orthogonal_vectors(self):
        """Given: 직교하는 두 벡터가 있을 때
        When: _calculate_similarity를 호출하면
        Then: 0.0의 유사도를 반환한다
        """
        # Given
        vector1 = Vector([1.0, 0.0])
        vector2 = Vector([0.0, 1.0])

        # When
        similarity = self.vector_store._calculate_similarity(vector1, vector2)

        # Then
        self.assertAlmostEqual(similarity, 0.0, places=5)

    def test_calculate_similarity_different_dimensions(self):
        """Given: 차원이 다른 두 벡터가 있을 때
        When: _calculate_similarity를 호출하면
        Then: 0.0을 반환한다
        """
        # Given
        vector1 = Vector([1.0, 2.0])
        vector2 = Vector([1.0, 2.0, 3.0])

        # When
        similarity = self.vector_store._calculate_similarity(vector1, vector2)

        # Then
        self.assertEqual(similarity, 0.0)

    async def test_load_config_success(self):
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

    async def test_load_config_not_found(self):
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
