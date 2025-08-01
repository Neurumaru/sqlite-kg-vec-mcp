"""
HNSW 인덱스 어댑터의 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.hnsw import HNSWBackend, HNSWIndex


class TestHNSWBackend(unittest.TestCase):
    """HNSWBackend Enum의 단위 테스트."""

    def test_enum_values(self):
        """Given: HNSWBackend Enum
        When: 값들을 확인하면
        Then: 예상된 값들이 정의되어 있어야 한다
        """
        # Given & When & Then
        self.assertEqual(HNSWBackend.HNSWLIB.value, "hnswlib")


class TestHNSWIndex(unittest.TestCase):
    """HNSWIndex 클래스의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    def test_init_default_parameters(self):
        """Given: 기본 매개변수로 HNSWIndex 초기화
        When: 인스턴스를 생성하면
        Then: 기본값들이 설정되어야 한다
        """
        # Given & When
        index = HNSWIndex()

        # Then
        self.assertEqual(index.space, "cosine")
        self.assertEqual(index.dim, 128)
        self.assertEqual(index.ef_construction, 200)
        self.assertEqual(index.m_parameter, 16)
        self.assertIsNone(index.index_dir)
        self.assertEqual(index.backend, HNSWBackend.HNSWLIB)
        self.assertFalse(index.is_initialized)

    def test_init_custom_parameters(self):
        """Given: 사용자 정의 매개변수
        When: HNSWIndex를 초기화하면
        Then: 사용자 정의 값들이 설정되어야 한다
        """
        # Given
        space = "l2"
        dim = 256
        ef_construction = 300
        m_parameter = 32
        backend = "hnswlib"

        # When
        index = HNSWIndex(
            space=space,
            dim=dim,
            ef_construction=ef_construction,
            m_parameter=m_parameter,
            index_dir=self.index_dir,
            backend=backend,
        )

        # Then
        self.assertEqual(index.space, space)
        self.assertEqual(index.dim, dim)
        self.assertEqual(index.ef_construction, ef_construction)
        self.assertEqual(index.m_parameter, m_parameter)
        self.assertEqual(index.index_dir, self.index_dir)
        self.assertEqual(index.backend, HNSWBackend.HNSWLIB)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_init_backend(self, mock_hnswlib_index):
        """Given: HNSWIndex 인스턴스
        When: _init_backend()를 호출하면
        Then: hnswlib.Index가 초기화되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex(space="cosine", dim=128)

        # When (이미 __init__에서 호출됨)
        # Then
        mock_hnswlib_index.assert_called_with(space="cosine", dim=128)
        self.assertEqual(index.index, mock_index_instance)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_init_index_success(self, mock_hnswlib_index):
        """Given: 초기화되지 않은 HNSWIndex
        When: init_index()를 호출하면
        Then: 인덱스가 초기화되고 설정되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        max_elements = 1000

        # When
        index.init_index(max_elements)

        # Then
        mock_index_instance.init_index.assert_called_once_with(
            max_elements=max_elements, ef_construction=200, M=16
        )
        mock_index_instance.set_ef.assert_called_once_with(200)
        self.assertTrue(index.is_initialized)
        self.assertEqual(index.current_capacity, max_elements)
        self.assertEqual(index.current_size, 0)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_add_item_success(self, mock_hnswlib_index):
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
    def test_add_item_not_initialized(self, mock_hnswlib_index):
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
    def test_add_item_replace_existing(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_add_items_batch_success(self, mock_hnswlib_index):
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
    def test_add_items_batch_empty(self, mock_hnswlib_index):
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
    def test_add_items_batch_mismatched_lengths(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_remove_item_success(self, mock_hnswlib_index):
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
    def test_remove_item_not_found(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_search_success(self, mock_hnswlib_index):
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
    def test_search_with_filter(self, mock_hnswlib_index):
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
    def test_search_empty_index(self, mock_hnswlib_index):
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
    def test_search_not_initialized(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_resize_index(self, mock_hnswlib_index):
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
    def test_resize_index_smaller_size(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    @patch.object(Path, "exists")
    def test_load_index_success(self, mock_path_exists, mock_hnswlib_index):
        """Given: 저장된 인덱스 파일들
        When: load_index()를 호출하면
        Then: 인덱스와 매핑이 로드되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_index_instance.get_max_elements.return_value = 1000
        mock_index_instance.get_current_count.return_value = 5
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex(index_dir=self.index_dir)

        # 파일 존재 시뮬레이션
        mock_path_exists.return_value = True

        # pickle 데이터 모킹
        mock_mappings = {
            "id_to_idx": {("node", 1): 0, ("node", 2): 1},
            "idx_to_id": {0: ("node", 1), 1: ("node", 2)},
        }

        with patch("builtins.open", create=True):
            with patch("pickle.load", return_value=mock_mappings):
                # When
                index.load_index()

                # Then
                mock_index_instance.load_index.assert_called_once()
                self.assertTrue(index.is_initialized)
                self.assertEqual(index.current_capacity, 1000)
                self.assertEqual(index.current_size, 5)
                self.assertEqual(index.id_to_idx, mock_mappings["id_to_idx"])
                self.assertEqual(index.idx_to_id, mock_mappings["idx_to_id"])

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    @patch.object(Path, "exists")
    def test_load_index_file_not_found(self, mock_path_exists, mock_hnswlib_index):
        """Given: 존재하지 않는 인덱스 파일
        When: load_index()를 호출하면
        Then: FileNotFoundError가 발생해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex(index_dir=self.index_dir)

        # 파일 존재하지 않음 시뮬레이션
        mock_path_exists.return_value = False

        # When & Then
        with self.assertRaises(FileNotFoundError):
            index.load_index()

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    @patch.object(Path, "mkdir")
    def test_save_index_success(self, mock_mkdir, mock_hnswlib_index):
        """Given: 초기화된 인덱스
        When: save_index()를 호출하면
        Then: 인덱스와 매핑이 저장되어야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex(index_dir=self.index_dir)
        index.init_index(1000)

        # 매핑 데이터 설정
        index.id_to_idx = {("node", 1): 0, ("node", 2): 1}
        index.idx_to_id = {0: ("node", 1), 1: ("node", 2)}

        with patch("builtins.open", create=True):
            with patch("pickle.dump") as mock_pickle_dump:
                # When
                index.save_index()

                # Then
                mock_index_instance.save_index.assert_called_once()
                mock_pickle_dump.assert_called_once()

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_save_index_not_initialized(self, mock_hnswlib_index):
        """Given: 초기화되지 않은 인덱스
        When: save_index()를 호출하면
        Then: RuntimeError가 발생해야 한다
        """
        # Given
        mock_index_instance = Mock()
        mock_hnswlib_index.return_value = mock_index_instance

        index = HNSWIndex()
        # init_index() 호출하지 않음

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            index.save_index()

        self.assertIn("Index is not initialized", str(context.exception))

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_build_from_embeddings(self, mock_hnswlib_index):
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

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_sync_with_outbox(self, mock_hnswlib_index):
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
