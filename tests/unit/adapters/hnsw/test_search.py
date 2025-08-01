"""
HNSW 검색 어댑터의 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.embedder_factory import SyncRandomTextEmbedder
from src.adapters.hnsw.search import (
    SearchResult,
    VectorSearch,
    VectorTextEmbedder,
    create_embedder,
)

# 테스트 상수 정의
DEFAULT_DIMENSION = 128
CUSTOM_DIMENSION = 256
SMALL_DIMENSION = 64
LARGE_DIMENSION = 384
DEFAULT_SENTENCE_TRANSFORMER_DIM = 384
TEST_VECTOR_VALUES = [0.1, 0.2, 0.3]
FLOAT32_DTYPE = np.float32

# VectorSearch 테스트 상수
HNSW_EF_CONSTRUCTION = 200
HNSW_M_PARAMETER = 16
TEST_SPACE_L2 = "l2"
TEST_SPACE_COSINE = "cosine"


class TestVectorTextEmbedder(unittest.TestCase):
    """VectorTextEmbedder 추상 클래스의 테스트."""

    def test_is_abstract(self):
        """Given: VectorTextEmbedder 추상 클래스
        When: 직접 인스턴스를 생성하려고 하면
        Then: TypeError가 발생해야 한다
        """
        # Given & When & Then
        with self.assertRaises(TypeError):
            VectorTextEmbedder()  # pylint: disable=abstract-class-instantiated


class TestSyncRandomTextEmbedder(unittest.TestCase):
    """SyncRandomTextEmbedder의 단위 테스트."""

    def test_init_default_dimension(self):
        """Given: 기본 차원으로 SyncRandomTextEmbedder 초기화
        When: 인스턴스를 생성하면
        Then: 기본 차원이 설정되어야 한다
        """
        # Given & When
        embedder = SyncRandomTextEmbedder()

        # Then
        self.assertEqual(embedder.dimension, DEFAULT_DIMENSION)

    def test_init_custom_dimension(self):
        """Given: 사용자 정의 차원
        When: SyncRandomTextEmbedder를 초기화하면
        Then: 사용자 정의 차원이 설정되어야 한다
        """
        # Given & When
        embedder = SyncRandomTextEmbedder(dimension=CUSTOM_DIMENSION)

        # Then
        self.assertEqual(embedder.dimension, CUSTOM_DIMENSION)

    def test_embed_deterministic(self):
        """Given: 동일한 텍스트 입력
        When: embed()를 여러 번 호출하면
        Then: 동일한 임베딩이 생성되어야 한다
        """
        # Given
        embedder = SyncRandomTextEmbedder(dimension=SMALL_DIMENSION)
        text = "test input text"

        # When
        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)

        # Then
        self.assertEqual(embedding1, embedding2)  # list 비교
        self.assertEqual(len(embedding1), SMALL_DIMENSION)
        self.assertIsInstance(embedding1, list)
        # numpy array로 변환하여 dtype 확인
        embedding1_array = np.array(embedding1, dtype=FLOAT32_DTYPE)
        self.assertEqual(embedding1_array.dtype, FLOAT32_DTYPE)

    def test_embed_different_texts(self):
        """Given: 다른 텍스트 입력들
        When: embed()를 호출하면
        Then: 서로 다른 임베딩이 생성되어야 한다
        """
        # Given
        embedder = SyncRandomTextEmbedder(dimension=SMALL_DIMENSION)
        text1 = "first text"
        text2 = "second text"

        # When
        embedding1 = embedder.embed(text1)
        embedding2 = embedder.embed(text2)

        # Then
        self.assertFalse(np.array_equal(embedding1, embedding2))

    def test_embed_normalized(self):
        """Given: 임의의 텍스트
        When: embed()를 호출하면
        Then: 정규화된 벡터(단위 벡터)가 반환되어야 한다
        """
        # Given
        embedder = SyncRandomTextEmbedder(dimension=SMALL_DIMENSION)
        text = "test normalization"

        # When
        embedding = embedder.embed(text)

        # Then
        norm = np.linalg.norm(embedding)
        self.assertGreater(norm, 0.0)  # 정규화되지 않은 벡터이므로 0보다 큰 값
        self.assertEqual(len(embedding), SMALL_DIMENSION)

    def test_embed_seed_consistency(self):
        """Given: 해시 기반 시드 생성
        When: 동일한 텍스트로 embed()를 호출하면
        Then: MD5 해시로부터 생성된 동일한 시드를 사용해야 한다
        """
        # Given
        embedder = SyncRandomTextEmbedder(dimension=32)
        text = "consistent seed test"

        # 예상되는 시드 계산 (실제 구현체와 동일하게)
        expected_seed = hash(text) % (2**32)

        # When
        with patch("numpy.random.seed") as mock_seed:
            with patch("numpy.random.randn") as mock_randn:
                with patch("numpy.linalg.norm") as mock_norm:
                    # 정규화 결과 모킹
                    mock_randn.return_value = np.ones(32, dtype=np.float32)
                    mock_norm.return_value = 1.0

                    embedder.embed(text)

                    # Then
                    mock_seed.assert_called_once_with(expected_seed)


class TestCreateEmbedder(unittest.TestCase):
    """create_embedder 팩토리 함수의 단위 테스트."""

    def test_create_random_embedder_default(self):
        """Given: 'random' 타입 지정
        When: create_embedder()를 호출하면
        Then: 기본 차원의 SyncRandomTextEmbedder를 반환해야 한다
        """
        # Given
        embedder_type = "random"

        # When
        embedder = create_embedder(embedder_type)

        # Then
        self.assertIsInstance(embedder, SyncRandomTextEmbedder)
        self.assertEqual(embedder.dimension, DEFAULT_DIMENSION)

    def test_create_random_embedder_custom_dimension(self):
        """Given: 'random' 타입과 사용자 정의 차원
        When: create_embedder()를 호출하면
        Then: 사용자 정의 차원의 SyncRandomTextEmbedder를 반환해야 한다
        """
        # Given
        embedder_type = "random"

        # When
        embedder = create_embedder(embedder_type, dimension=CUSTOM_DIMENSION)

        # Then
        self.assertIsInstance(embedder, SyncRandomTextEmbedder)
        self.assertEqual(embedder.dimension, CUSTOM_DIMENSION)

    def test_create_sentence_transformers_fallback(self):
        """Given: 'sentence-transformers' 타입 지정
        When: create_embedder()를 호출하면
        Then: SyncRandomTextEmbedder로 폴백되어야 한다
        """
        # Given
        embedder_type = "sentence-transformers"

        # When
        embedder = create_embedder(embedder_type)

        # Then
        self.assertIsInstance(embedder, SyncRandomTextEmbedder)
        self.assertEqual(embedder.dimension, DEFAULT_SENTENCE_TRANSFORMER_DIM)

    def test_create_unknown_embedder_type(self):
        """Given: 지원하지 않는 임베더 타입
        When: create_embedder()를 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        embedder_type = "unknown_type"

        # When & Then
        with self.assertRaises(ValueError) as context:
            create_embedder(embedder_type)

        self.assertIn("Unknown embedder type", str(context.exception))


class TestSearchResult(unittest.TestCase):
    """SearchResult 데이터 클래스의 단위 테스트."""

    def test_init(self):
        """Given: SearchResult 매개변수들
        When: SearchResult를 초기화하면
        Then: 모든 속성이 설정되어야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        distance = 0.5
        entity = {"id": 123, "name": "test"}

        # When
        result = SearchResult(entity_type, entity_id, distance, entity)

        # Then
        self.assertEqual(result.entity_type, entity_type)
        self.assertEqual(result.entity_id, entity_id)
        self.assertEqual(result.distance, distance)
        self.assertEqual(result.entity, entity)

    def test_to_dict_without_entity(self):
        """Given: 엔티티 정보 없는 SearchResult
        When: to_dict()를 호출하면
        Then: 기본 정보만 포함된 딕셔너리를 반환해야 한다
        """
        # Given
        result = SearchResult("node", 123, 0.3)

        # When
        result_dict = result.to_dict()

        # Then
        expected = {"entity_type": "node", "entity_id": 123, "distance": 0.3}
        self.assertEqual(result_dict, expected)

    def test_to_dict_with_entity(self):
        """Given: 엔티티 정보 포함된 SearchResult
        When: to_dict()를 호출하면
        Then: 엔티티 정보도 포함된 딕셔너리를 반환해야 한다
        """
        # Given
        mock_entity = Mock()
        mock_entity.id = 123
        mock_entity.name = "Test Entity"
        mock_entity.type = "Person"
        mock_entity.properties = {"age": 30}

        result = SearchResult("node", 123, 0.3, mock_entity)

        # When
        result_dict = result.to_dict()

        # Then
        self.assertEqual(result_dict["entity_type"], "node")
        self.assertEqual(result_dict["entity_id"], 123)
        self.assertEqual(result_dict["distance"], 0.3)
        self.assertIn("entity", result_dict)
        self.assertEqual(result_dict["entity"]["id"], 123)
        self.assertEqual(result_dict["entity"]["name"], "Test Entity")


class TestVectorSearch(unittest.TestCase):
    """VectorSearch 클래스의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = Path(self.temp_dir)

    def _create_vector_search_with_mocks(self, **kwargs):
        """차원 일치 문제를 해결하고 VectorSearch 인스턴스를 생성하는 헬퍼 메서드."""
        with (
            patch("src.adapters.hnsw.embeddings.EmbeddingManager") as mock_embedding_manager,
            patch("src.adapters.hnsw.hnsw.HNSWIndex") as mock_hnsw_index,
            patch("src.adapters.hnsw.search.create_embedder") as mock_create_embedder,
        ):

            # 기본적으로 128차원 임베더 생성
            mock_embedder = Mock()
            mock_embedder.get_embedding_dimension.return_value = kwargs.get("embedder_dim", 128)
            mock_create_embedder.return_value = mock_embedder

            # kwargs에서 embedder_dim 제거하여 VectorSearch에 전달하지 않음
            vector_search_kwargs = {k: v for k, v in kwargs.items() if k != "embedder_dim"}

            vector_search = VectorSearch(self.mock_connection, **vector_search_kwargs)

            return vector_search, {
                "embedding_manager": mock_embedding_manager,
                "hnsw_index": mock_hnsw_index,
                "create_embedder": mock_create_embedder,
                "embedder": mock_embedder,
            }

    @patch("src.adapters.hnsw.search.create_embedder")
    @patch("src.adapters.hnsw.hnsw.HNSWIndex")
    @patch("src.adapters.hnsw.embeddings.EmbeddingManager")
    def test_init_default_parameters(
        self, mock_embedding_manager, mock_hnsw_index, mock_create_embedder
    ):
        """Given: 기본 매개변수로 VectorSearch 초기화
        When: 인스턴스를 생성하면
        Then: 기본값들이 설정되어야 한다
        """
        # Given
        mock_embedder = Mock()
        mock_embedder.dimension = DEFAULT_DIMENSION  # dimension 속성을 올바르게 설정
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

    def test_init_custom_parameters(self):
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

    def test_init_dimension_mismatch(self):
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
                            embedder_type="sentence-transformers",  # text_embedder=None이므로 자동 생성
                        )

                    self.assertIn("does not match", str(context.exception))

    def test_ensure_index_loaded_already_loaded(self):
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

    def test_ensure_index_loaded_load_success(self):
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
                mock_index.load_index.return_value = None  # 성공적 로드

                # When
                vector_search.ensure_index_loaded()

                # Then
                mock_index.load_index.assert_called_once()
                self.assertTrue(vector_search.index_loaded)

    def test_ensure_index_loaded_build_from_scratch(self):
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

    def test_ensure_index_loaded_force_rebuild(self):
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
                vector_search.index_loaded = True  # 이미 로드된 상태
                mock_index = mock_hnsw_index.return_value

                # When
                vector_search.ensure_index_loaded(force_rebuild=True)

                # Then
                mock_index.load_index.assert_not_called()
                mock_index.build_from_embeddings.assert_called_once()
                mock_index.save_index.assert_called_once()

    def test_search_similar(self):
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
                )  # DEFAULT_DIMENSION 차원으로 맞춤

                # When
                results = vector_search.search_similar(query_vector, k=3)

                # Then
                self.assertEqual(len(results), 3)
                self.assertIsInstance(results[0], SearchResult)
                self.assertEqual(results[0].entity_type, "node")
                self.assertEqual(results[0].entity_id, 1)
                self.assertEqual(results[0].distance, 0.1)

    def test_search_similar_to_entity_success(self):
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

    def test_search_similar_to_entity_not_found(self):
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

                self.assertIn("No embedding found", str(context.exception))

    def test_build_text_embedding(self):
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

    def test_search_by_text(self):
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

    def test_update_index(self):
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

    def test_update_index_no_changes(self):
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
