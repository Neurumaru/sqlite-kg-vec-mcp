"""
HNSW 인덱스 검색 메서드 테스트.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.hnsw import HNSWIndex


class TestHNSWIndexSearch(unittest.TestCase):

    """HNSWIndex.search 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 초기화된 인덱스와 쿼리 벡터
        When: search()를 호출하면
        Then: 검색 결과를 반환해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        # 검색 결과 모킹
        mock_indices = np.array([[0, 1, 2]])
        mock_distances = np.array([[0.1, 0.2, 0.3]])
        mock_index_instance.knn_query.return_value = (mock_indices, mock_distances)

        index = HNSWIndex()
        index.init_index(1000)

        # 아이템들 추가 (매핑 설정)
        index.id_to_idx = {("node", 1): 0, ("node", 2): 1, ("edge", 3): 2}
        index.idx_to_id = {0: ("node", 1), 1: ("node", 2), 2: ("edge", 3)}
        index.current_size = 3

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # When
        results = index.search(query_vector, k=3)

        # Then
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], ("node", 1, 0.1))
        self.assertEqual(results[1], ("node", 2, 0.2))
        self.assertEqual(results[2], ("edge", 3, 0.3))
        # numpy 배열 비교를 위해 call_args를 직접 확인
        call_args = mock_index_instance.knn_query.call_args
        np.testing.assert_array_equal(call_args[0][0], query_vector)
        self.assertEqual(call_args[1]["k"], 3)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success_when_filtered(self, mock_hnswlib_index):
        """Given: 엔티티 타입 필터
        When: search()를 호출하면
        Then: 필터된 결과만 반환해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        # 검색 결과 모킹
        mock_indices = np.array([[0, 1, 2]])
        mock_distances = np.array([[0.1, 0.2, 0.3]])
        mock_index_instance.knn_query.return_value = (mock_indices, mock_distances)

        index = HNSWIndex()
        index.init_index(1000)

        # 아이템들 추가 (매핑 설정)
        index.id_to_idx = {("node", 1): 0, ("node", 2): 1, ("edge", 3): 2}
        index.idx_to_id = {0: ("node", 1), 1: ("node", 2), 2: ("edge", 3)}
        index.current_size = 3

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # When - 노드만 필터링
        results = index.search(query_vector, k=3, filter_entity_types=["node"])

        # Then - 노드 타입만 반환되어야 함
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ("node", 1, 0.1))
        self.assertEqual(results[1], ("node", 2, 0.2))

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_returns_empty_when_empty_index(self, mock_hnswlib_index):
        """Given: 빈 인덱스
        When: search()를 호출하면
        Then: 빈 결과를 반환해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)
        # current_size는 0으로 유지

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # When
        results = index.search(query_vector, k=3)

        # Then
        self.assertEqual(results, [])

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_runtime_error_when_not_initialized(self, mock_hnswlib_index):
        """Given: 초기화되지 않은 인덱스
        When: search()를 호출하면
        Then: RuntimeError가 발생해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        # init_index() 호출하지 않음

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            index.search(query_vector)

        self.assertIn("Index is not initialized", str(context.exception))


if __name__ == "__main__":
    unittest.main()
