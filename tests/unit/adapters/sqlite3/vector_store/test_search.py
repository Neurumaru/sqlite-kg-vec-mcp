"""
SQLiteVectorStore 검색 및 유사도 기능 테스트.
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


class TestSQLiteVectorStoreSearch(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore 검색 및 유사도 기능 테스트."""

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

    async def test_success_when_search_similar_basic(self):
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

    async def test_success_when_search_similar_with_filters(self):
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

    async def test_success_when_search_similar_with_vectors(self):
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

    async def test_success_when_search_by_ids(self):
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

    async def test_success_when_batch_search(self):
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

    async def test_success_when_search_by_metadata(self):
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

    def test_success_when_calculate_similarity_identical_vectors(self):
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

    def test_success_when_calculate_similarity_orthogonal_vectors(self):
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

    def test_zero_when_calculate_similarity_different_dimensions(self):
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


class TestSQLiteVectorStoreMetadata(unittest.IsolatedAsyncioTestCase):
    """SQLiteVectorStore 메타데이터 기능 테스트."""

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

    async def test_success_when_get_metadata(self):
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

    async def test_none_when_get_metadata_not_found(self):
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

    async def test_success_when_update_metadata(self):
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


if __name__ == "__main__":
    unittest.main()
