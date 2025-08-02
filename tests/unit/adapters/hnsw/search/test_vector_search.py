"""
HNSW 검색 어댑터의 VectorSearch 관련 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.embedder_factory import SyncRandomTextEmbedder
from src.adapters.hnsw.search import SearchResult, VectorSearch

# 테스트 상수 정의
DEFAULT_DIMENSION = 128
CUSTOM_DIMENSION = 256
FLOAT32_DTYPE = np.float32
TEST_VECTOR_VALUES = [0.1, 0.2, 0.3]

# VectorSearch 테스트 상수
HNSW_EF_CONSTRUCTION = 200
HNSW_M_PARAMETER = 16
TEST_SPACE_L2 = "l2"
TEST_SPACE_COSINE = "cosine"


class TestVectorSearchInit(unittest.TestCase):
    """VectorSearch.__init__ 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    @patch("src.adapters.hnsw.search.create_embedder")
    @patch("src.adapters.hnsw.hnsw.HNSWIndex")
    @patch("src.adapters.hnsw.embeddings.EmbeddingManager")
    def test_success(self, mock_embedding_manager, mock_hnsw_index, mock_create_embedder):
        """Given: 기본 매개변수로 VectorSearch 초기화
        When: 인스턴스를 생성하면
        Then: 기본값들이 설정되어야 한다
        """
        # Given
        mock_embedder = Mock()
        mock_embedder.dimension = DEFAULT_DIMENSION
        mock_create_embedder.return_value = mock_embedder

        # When
        vector_search = VectorSearch(
            self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
        )

        # Then
        mock_embedding_manager.assert_called_once_with(self.mock_connection)
        mock_hnsw_index.assert_called_once()
        self.assertFalse(vector_search.index_loaded)
        self.assertIsNotNone(vector_search.text_embedder)

    def test_success_when_custom_parameters(self):
        """Given: 사용자 정의 매개변수
        When: VectorSearch를 초기화하면
        Then: 사용자 정의 값들이 설정되어야 한다
        """
        # Given
        custom_embedder = SyncRandomTextEmbedder(dimension=CUSTOM_DIMENSION)

        # When
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection,
                    index_dir=str(self.index_dir),
                    embedding_dim=CUSTOM_DIMENSION,
                    space=TEST_SPACE_L2,
                    text_embedder=custom_embedder,
                )

                # Then
                mock_hnsw_index.assert_called_once_with(
                    space=TEST_SPACE_L2,
                    dim=CUSTOM_DIMENSION,
                    ef_construction=HNSW_EF_CONSTRUCTION,
                    m_parameter=HNSW_M_PARAMETER,
                    index_dir=str(self.index_dir),
                )
                self.assertEqual(vector_search.text_embedder, custom_embedder)

    def test_value_error_when_dimension_mismatch(self):
        """Given: 자동 생성된 임베더의 차원과 인덱스 차원이 다를 때
        When: VectorSearch를 초기화하면
        Then: ValueError가 발생해야 한다
        """
        # Given & When & Then
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex"):
                with patch("src.adapters.hnsw.search.create_embedder") as mock_create_embedder:
                    # 차원이 다른 임베더를 반환하도록 모킹
                    mock_embedder = Mock()
                    mock_embedder.dimension = CUSTOM_DIMENSION  # 다른 차원
                    mock_create_embedder.return_value = mock_embedder

                    with self.assertRaises(ValueError) as context:
                        VectorSearch(
                            self.mock_connection,
                            embedding_dim=DEFAULT_DIMENSION,
                            embedder_type="sentence-transformers",
                        )

                    self.assertIn("일치하지 않습니다", str(context.exception))


class TestVectorSearchEnsureIndexLoaded(unittest.TestCase):
    """VectorSearch.ensure_index_loaded 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success_when_already_loaded(self):
        """Given: 이미 로드된 인덱스
        When: ensure_index_loaded()를 호출하면
        Then: 추가 작업을 하지 않아야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                vector_search.index_loaded = True
                mock_index = mock_hnsw_index.return_value

                # When
                vector_search.ensure_index_loaded()

                # Then
                mock_index.load_index.assert_not_called()
                mock_index.build_from_embeddings.assert_not_called()

    def test_success_when_load_success(self):
        """Given: 저장된 인덱스 파일이 있을 때
        When: ensure_index_loaded()를 호출하면
        Then: 인덱스를 로드해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value
                mock_index.load_index.return_value = None

                # When
                vector_search.ensure_index_loaded()

                # Then
                mock_index.load_index.assert_called_once()
                self.assertTrue(vector_search.index_loaded)

    def test_success_when_build_from_scratch(self):
        """Given: 저장된 인덱스 파일이 없을 때
        When: ensure_index_loaded()를 호출하면
        Then: 임베딩으로부터 인덱스를 구축해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager") as mock_embedding_manager:
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value
                mock_index.load_index.side_effect = FileNotFoundError("Index not found")

                # When
                vector_search.ensure_index_loaded()

                # Then
                mock_index.build_from_embeddings.assert_called_once_with(
                    mock_embedding_manager.return_value
                )
                mock_index.save_index.assert_called_once()
                self.assertTrue(vector_search.index_loaded)

    def test_success_when_force_rebuild(self):
        """Given: force_rebuild=True
        When: ensure_index_loaded()를 호출하면
        Then: 기존 인덱스를 무시하고 재구축해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                vector_search.index_loaded = True
                mock_index = mock_hnsw_index.return_value

                # When
                vector_search.ensure_index_loaded(force_rebuild=True)

                # Then
                mock_index.load_index.assert_not_called()
                mock_index.build_from_embeddings.assert_called_once()
                mock_index.save_index.assert_called_once()


class TestVectorSearchSearchSimilar(unittest.TestCase):
    """VectorSearch.search_similar 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success(self):
        """Given: 쿼리 벡터와 검색 매개변수
        When: search_similar()을 호출하면
        Then: SearchResult 목록을 반환해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value

                # 검색 결과 모킹
                mock_search_results = [("node", 1, 0.1), ("edge", 2, 0.2), ("node", 3, 0.3)]
                mock_index.search.return_value = mock_search_results

                query_vector = np.array(
                    TEST_VECTOR_VALUES + [0.0] * (DEFAULT_DIMENSION - len(TEST_VECTOR_VALUES)),
                    dtype=FLOAT32_DTYPE,
                )

                # When
                results = vector_search.search_similar(query_vector, k=3)

                # Then
                self.assertEqual(len(results), 3)
                self.assertIsInstance(results[0], SearchResult)
                self.assertEqual(results[0].entity_type, "node")
                self.assertEqual(results[0].entity_id, 1)
                self.assertEqual(results[0].distance, 0.1)


class TestVectorSearchSearchSimilarToEntity(unittest.TestCase):
    """VectorSearch.search_similar_to_entity 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success(self):
        """Given: 존재하는 엔티티
        When: search_similar_to_entity()를 호출하면
        Then: 해당 엔티티와 유사한 엔티티들을 반환해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager") as mock_embedding_manager:
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value

                # 임베딩 모킹
                mock_embedding = Mock()
                mock_embedding.embedding = np.array(
                    TEST_VECTOR_VALUES + [0.0] * (DEFAULT_DIMENSION - len(TEST_VECTOR_VALUES)),
                    dtype=FLOAT32_DTYPE,
                )
                mock_embedding_manager.return_value.get_embedding.return_value = mock_embedding

                # 검색 결과 모킹 (첫 번째는 자기 자신, 나머지는 유사한 엔티티들)
                mock_search_results = [
                    ("node", 1, 0.0),  # 자기 자신
                    ("node", 2, 0.1),
                    ("edge", 3, 0.2),
                ]
                mock_index.search.return_value = mock_search_results

                # When
                results = vector_search.search_similar_to_entity("node", 1, k=2)

                # Then
                self.assertEqual(len(results), 2)  # 자기 자신 제외
                self.assertEqual(results[0].entity_type, "node")
                self.assertEqual(results[0].entity_id, 2)
                self.assertEqual(results[1].entity_type, "edge")
                self.assertEqual(results[1].entity_id, 3)

    def test_value_error_when_entity_not_found(self):
        """Given: 존재하지 않는 엔티티
        When: search_similar_to_entity()를 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager") as mock_embedding_manager:
            with patch("src.adapters.hnsw.hnsw.HNSWIndex"):
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )

                # 임베딩 없음 모킹
                mock_embedding_manager.return_value.get_embedding.return_value = None

                # When & Then
                with self.assertRaises(ValueError) as context:
                    vector_search.search_similar_to_entity("node", 999, k=2)

                self.assertIn("임베딩을 찾을 수 없습니다", str(context.exception))


class TestVectorSearchBuildTextEmbedding(unittest.TestCase):
    """VectorSearch.build_text_embedding 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success(self):
        """Given: 텍스트 입력
        When: build_text_embedding()을 호출하면
        Then: 임베딩 벡터를 반환해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex"):
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                text = "test text input"

                # When
                embedding = vector_search.build_text_embedding(text)

                # Then
                self.assertIsInstance(embedding, np.ndarray)
                self.assertEqual(embedding.dtype, np.float32)


class TestVectorSearchSearchByText(unittest.TestCase):
    """VectorSearch.search_by_text 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success(self):
        """Given: 텍스트 쿼리
        When: search_by_text()를 호출하면
        Then: 텍스트와 유사한 엔티티들을 반환해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value

                # 검색 결과 모킹
                mock_search_results = [("node", 1, 0.1), ("node", 2, 0.2)]
                mock_index.search.return_value = mock_search_results

                query_text = "search query text"

                # When
                results = vector_search.search_by_text(query_text, k=2)

                # Then
                self.assertEqual(len(results), 2)
                self.assertIsInstance(results[0], SearchResult)
                mock_index.search.assert_called_once()


class TestVectorSearchUpdateIndex(unittest.TestCase):
    """VectorSearch.update_index 메서드의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)

    def test_success(self):
        """Given: 아웃박스에 대기 중인 변경사항들
        When: update_index()를 호출하면
        Then: 인덱스가 업데이트되고 변경사항 수를 반환해야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value
                mock_index.sync_with_outbox.return_value = 5

                # When
                count = vector_search.update_index(batch_size=50)

                # Then
                self.assertEqual(count, 5)
                mock_index.sync_with_outbox.assert_called_once_with(
                    vector_search.embedding_manager, 50
                )
                mock_index.save_index.assert_called_once()

    def test_success_when_no_changes(self):
        """Given: 아웃박스에 변경사항이 없을 때
        When: update_index()를 호출하면
        Then: 인덱스를 저장하지 않아야 한다
        """
        # Given
        with patch("src.adapters.hnsw.embeddings.EmbeddingManager"):
            with patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index:
                vector_search = VectorSearch(
                    self.mock_connection, embedding_dim=DEFAULT_DIMENSION, embedder_type="random"
                )
                mock_index = mock_hnsw_index.return_value
                mock_index.sync_with_outbox.return_value = 0  # 변경사항 없음

                # When
                count = vector_search.update_index()

                # Then
                self.assertEqual(count, 0)
                mock_index.save_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
