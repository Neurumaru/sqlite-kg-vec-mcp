"""
HNSW 인덱스 관리 메서드 테스트.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.hnsw import HNSWIndex


class TestHNSWIndexAddItem(unittest.TestCase):
    """HNSWIndex.add_item 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 초기화된 인덱스와 벡터
        When: add_item()을 호출하면
        Then: 아이템이 추가되고 인덱스가 반환되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        entity_type = "node"
        entity_id = 123
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # When
        result_idx = index.add_item(entity_type, entity_id, vector)

        # Then
        self.assertEqual(result_idx, 0)
        mock_index_instance.add_items.assert_called_once()
        self.assertEqual(index.current_size, 1)
        self.assertIn((entity_type, entity_id), index.id_to_idx)
        self.assertIn(0, index.idx_to_id)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_runtime_error_when_not_initialized(self, mock_hnswlib_index):
        """Given: 초기화되지 않은 인덱스
        When: add_item()을 호출하면
        Then: RuntimeError가 발생해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        # init_index() 호출하지 않음

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            index.add_item("node", 123, np.array([0.1, 0.2, 0.3]))

        self.assertIn("Index is not initialized", str(context.exception))

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success_when_replace_existing(self, mock_hnswlib_index):
        """Given: 이미 존재하는 아이템과 replace_existing=True
        When: add_item()을 호출하면
        Then: 기존 아이템이 제거되고 새로운 아이템이 추가되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # 첫 번째 아이템 추가
        entity_type = "node"
        entity_id = 123
        vector1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        index.add_item(entity_type, entity_id, vector1)

        # 두 번째 벡터 (같은 엔티티, 다른 벡터)
        vector2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)

        # When
        result_idx = index.add_item(entity_type, entity_id, vector2, replace_existing=True)

        # Then
        # 새로운 인덱스가 반환되어야 함 (제거 후 추가)
        self.assertIsInstance(result_idx, int)
        # mark_deleted가 호출되었는지 확인
        mock_index_instance.mark_deleted.assert_called()


class TestHNSWIndexAddItemsBatch(unittest.TestCase):
    """HNSWIndex.add_items_batch 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 다수의 벡터들
        When: add_items_batch()를 호출하면
        Then: 모든 아이템들이 배치로 추가되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        entity_types = ["node", "node", "edge"]
        entity_ids = [1, 2, 3]
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)

        # When
        result_indices = index.add_items_batch(entity_types, entity_ids, vectors)

        # Then
        self.assertEqual(len(result_indices), 3)
        self.assertEqual(result_indices, [0, 1, 2])
        mock_index_instance.add_items.assert_called_once()
        self.assertEqual(index.current_size, 3)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success_when_empty(self, mock_hnswlib_index):
        """Given: 빈 벡터 배열
        When: add_items_batch()를 호출하면
        Then: 빈 리스트가 반환되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # When
        result = index.add_items_batch([], [], np.array([]))

        # Then
        self.assertEqual(result, [])

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_value_error_when_mismatched_lengths(self, mock_hnswlib_index):
        """Given: 길이가 다른 입력 배열들
        When: add_items_batch()를 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # When & Then
        with self.assertRaises(ValueError) as context:
            index.add_items_batch(
                ["node", "edge"], [1, 2, 3], np.array([[0.1, 0.2], [0.3, 0.4]])  # 길이가 다름
            )

        self.assertIn("must have the same length", str(context.exception))


class TestHNSWIndexRemoveItem(unittest.TestCase):
    """HNSWIndex.remove_item 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 존재하는 아이템
        When: remove_item()을 호출하면
        Then: 아이템이 제거되고 True가 반환되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # 아이템 추가
        entity_type = "node"
        entity_id = 123
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        index.add_item(entity_type, entity_id, vector)

        # When
        result = index.remove_item(entity_type, entity_id)

        # Then
        self.assertTrue(result)
        mock_index_instance.mark_deleted.assert_called_with(0)
        self.assertNotIn((entity_type, entity_id), index.id_to_idx)
        self.assertNotIn(0, index.idx_to_id)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_returns_false_when_not_found(self, mock_hnswlib_index):
        """Given: 존재하지 않는 아이템
        When: remove_item()을 호출하면
        Then: False가 반환되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # When
        result = index.remove_item("node", 999)

        # Then
        self.assertFalse(result)


class TestHNSWIndexResizeIndex(unittest.TestCase):
    """HNSWIndex.resize_index 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 현재 용량보다 큰 새로운 크기
        When: resize_index()를 호출하면
        Then: 인덱스가 리사이즈되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        new_size = 2000

        # When
        index.resize_index(new_size)

        # Then
        mock_index_instance.resize_index.assert_called_once_with(new_size)
        self.assertEqual(index.current_capacity, new_size)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_skipped_when_smaller_size(self, mock_hnswlib_index):
        """Given: 현재 용량보다 작은 새로운 크기
        When: resize_index()를 호출하면
        Then: 리사이즈하지 않아야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        smaller_size = 500

        # When
        index.resize_index(smaller_size)

        # Then
        mock_index_instance.resize_index.assert_not_called()
        self.assertEqual(index.current_capacity, 1000)  # 변경되지 않음


if __name__ == "__main__":
    unittest.main()
