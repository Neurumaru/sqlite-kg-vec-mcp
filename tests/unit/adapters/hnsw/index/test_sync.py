"""
HNSW 인덱스 동기화 메서드 테스트.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.hnsw import HNSWIndex


class TestHNSWIndexBuildFromEmbeddings(unittest.TestCase):
    """HNSWIndex.build_from_embeddings 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: EmbeddingManager와 임베딩 데이터
        When: build_from_embeddings()를 호출하면
        Then: 임베딩들로부터 인덱스가 구축되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()

        # Mock EmbeddingManager
        mock_embedding_manager = Mock()
        mock_embedding_manager.connection = Mock()
        mock_cursor = Mock()
        mock_embedding_manager.connection.cursor.return_value = mock_cursor

        # 전체 개수 쿼리 결과
        mock_cursor.fetchone.return_value = (5,)

        # Mock 임베딩 데이터
        mock_embedding1 = Mock()
        mock_embedding1.entity_id = 1
        mock_embedding1.embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_embedding2 = Mock()
        mock_embedding2.entity_id = 2
        mock_embedding2.embedding = np.array([0.4, 0.5, 0.6], dtype=np.float32)

        # get_all_embeddings 호출 시 첫 번째 배치와 빈 배치 반환
        mock_embedding_manager.get_all_embeddings.side_effect = [
            [mock_embedding1, mock_embedding2],  # 첫 번째 배치
            [],  # 두 번째 배치 (빈 결과)
        ]

        # When
        result = index.build_from_embeddings(
            mock_embedding_manager, entity_types=["node"], batch_size=10
        )

        # Then
        self.assertEqual(result, 2)
        self.assertTrue(index.is_initialized)
        # add_items_batch가 호출되었는지 확인
        mock_index_instance.add_items.assert_called()


class TestHNSWIndexSyncWithOutbox(unittest.TestCase):
    """HNSWIndex.sync_with_outbox 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
        """Given: 아웃박스의 벡터 연산들
        When: sync_with_outbox()를 호출하면
        Then: 연산들이 처리되고 인덱스가 동기화되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        index.init_index(1000)

        # Mock EmbeddingManager
        mock_embedding_manager = Mock()
        mock_embedding_manager.process_outbox.return_value = 2

        mock_cursor = Mock()
        mock_embedding_manager.connection = Mock()
        mock_embedding_manager.connection.cursor.return_value = mock_cursor

        # 완료된 연산들 모킹
        mock_operations = [
            {"operation_type": "insert", "entity_type": "node", "entity_id": 1},
            {"operation_type": "delete", "entity_type": "node", "entity_id": 2},
        ]
        mock_cursor.fetchall.return_value = mock_operations

        # 임베딩 조회 결과 모킹
        mock_embedding = Mock()
        mock_embedding.embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedding_manager.get_embedding.return_value = mock_embedding

        # When
        result = index.sync_with_outbox(mock_embedding_manager, batch_size=10)

        # Then
        self.assertEqual(result, 2)
        mock_embedding_manager.process_outbox.assert_called_once_with(10)
        # get_embedding이 insert 연산에 대해 호출되었는지 확인
        mock_embedding_manager.get_embedding.assert_called_with("node", 1)


if __name__ == "__main__":
    unittest.main()
