"""
HNSW 검색 어댑터의 Embedder 관련 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

import unittest
from unittest.mock import patch

import numpy as np

from src.adapters.hnsw.embedder_factory import SyncRandomTextEmbedder
from src.adapters.hnsw.search import VectorTextEmbedder, create_embedder

# 테스트 상수 정의
DEFAULT_DIMENSION = 128
CUSTOM_DIMENSION = 256
SMALL_DIMENSION = 64
DEFAULT_SENTENCE_TRANSFORMER_DIM = 384
FLOAT32_DTYPE = np.float32


class TestVectorTextEmbedderInstantiation(unittest.TestCase):
    """VectorTextEmbedder 추상 클래스 인스턴스화의 단위 테스트."""

    def test_type_error_when_abstract_instantiation(self):
        """Given: VectorTextEmbedder 추상 클래스
        When: 직접 인스턴스를 생성하려고 하면
        Then: TypeError가 발생해야 한다
        """
        # Given & When & Then
        with self.assertRaises(TypeError):
            VectorTextEmbedder()  # pylint: disable=abstract-class-instantiated


class TestSyncRandomTextEmbedderInit(unittest.TestCase):
    """SyncRandomTextEmbedder.__init__ 메서드의 단위 테스트."""

    def test_success(self):
        """Given: 기본 차원으로 SyncRandomTextEmbedder 초기화
        When: 인스턴스를 생성하면
        Then: 기본 차원이 설정되어야 한다
        """
        # Given & When
        embedder = SyncRandomTextEmbedder()

        # Then
        self.assertEqual(embedder.dimension, DEFAULT_DIMENSION)

    def test_success_when_custom_dimension(self):
        """Given: 사용자 정의 차원
        When: SyncRandomTextEmbedder를 초기화하면
        Then: 사용자 정의 차원이 설정되어야 한다
        """
        # Given & When
        embedder = SyncRandomTextEmbedder(dimension=CUSTOM_DIMENSION)

        # Then
        self.assertEqual(embedder.dimension, CUSTOM_DIMENSION)


class TestSyncRandomTextEmbedderEmbed(unittest.TestCase):
    """SyncRandomTextEmbedder.embed 메서드의 단위 테스트."""

    def test_success(self):
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

    def test_success_when_different_texts(self):
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

    def test_success_when_normalized(self):
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

    def test_success_when_seed_consistency(self):
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

    def test_success(self):
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

    def test_success_when_custom_dimension(self):
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

    def test_success_when_sentence_transformers_fallback(self):
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

    def test_value_error_when_unknown_embedder_type(self):
        """Given: 지원하지 않는 임베더 타입
        When: create_embedder()를 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        embedder_type = "unknown_type"

        # When & Then
        with self.assertRaises(ValueError) as context:
            create_embedder(embedder_type)

        self.assertIn("알 수 없는 임베더 유형", str(context.exception))


if __name__ == "__main__":
    unittest.main()
