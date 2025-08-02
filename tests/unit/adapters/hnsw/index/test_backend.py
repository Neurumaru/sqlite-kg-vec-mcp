"""
HNSW Backend 및 초기화 테스트.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.hnsw.hnsw import HNSWBackend, HNSWIndex


class TestHNSWBackend(unittest.TestCase):
    """HNSWBackend Enum의 단위 테스트."""

    def test_success(self):
        """Given: HNSWBackend Enum
        When: 값들을 확인하면
        Then: 예상된 값들이 정의되어 있어야 한다
        """
        # Given & When & Then
        self.assertEqual(HNSWBackend.HNSWLIB.value, "hnswlib")


class TestHNSWIndexInit(unittest.TestCase):
    """HNSWIndex.__init__ 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    def test_success_when_default_parameters(self):
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

    def test_success_when_custom_parameters(self):
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
    def test_success_when_init_backend(self, mock_hnswlib_index):
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


class TestHNSWIndexInitIndex(unittest.TestCase):
    """HNSWIndex.init_index 메서드의 단위 테스트."""

    @patch("src.adapters.hnsw.hnsw.hnswlib.Index")
    def test_success(self, mock_hnswlib_index):
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


if __name__ == "__main__":
    unittest.main()
