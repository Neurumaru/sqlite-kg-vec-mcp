"""
HNSW 인덱스 저장/로드 메서드 테스트.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.hnsw.hnsw import HNSWIndex


class TestHNSWIndexLoadIndex(unittest.TestCase):
    """HNSWIndex.load_index 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    @patch.object(Path, "exists")
    def test_success(self, mock_path_exists, mock_hnswlib_index):
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
    def test_file_not_found_error_when_file_not_found(self, mock_path_exists, mock_hnswlib_index):
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


class TestHNSWIndexSaveIndex(unittest.TestCase):
    """HNSWIndex.save_index 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    @patch.object(Path, "mkdir")
    def test_success(self, mock_mkdir, mock_hnswlib_index):
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
    def test_runtime_error_when_not_initialized(self, mock_hnswlib_index):
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


if __name__ == "__main__":
    unittest.main()
