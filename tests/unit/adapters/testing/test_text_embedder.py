"""
테스트 텍스트 임베딩 어댑터의 단위 테스트.

헥사고날 아키텍처 원칙에 따라 포트 인터페이스를 구현한 어댑터를 테스트합니다.
테스트 격리 원칙을 준수하여 실제 어댑터를 import하여 사용합니다.
"""

# pylint: disable=protected-access

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from src.adapters.testing.text_embedder import RandomTextEmbedder
from src.dto.embedding import EmbeddingResult
from src.ports.text_embedder import TextEmbedder


class TestRandomTextEmbedder(unittest.IsolatedAsyncioTestCase):
    """RandomTextEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.default_dimension = 128
        self.custom_dimension = 256
        self.test_text = "테스트 임베딩 텍스트"

        # 기본 차원과 커스텀 차원 어댑터 초기화
        self.embedder = RandomTextEmbedder(self.default_dimension)
        self.custom_embedder = RandomTextEmbedder(self.custom_dimension)

    # 초기화 테스트
    def test_initialization_with_default_dimension(self):
        """Given: 기본 차원으로 초기화
        When: 어댑터를 생성하면
        Then: 올바른 속성값들이 설정된다
        """
        # Given & When: setUp에서 수행됨

        # Then
        self.assertEqual(self.embedder._dimension, self.default_dimension)
        self.assertEqual(self.embedder.model_name, f"random-{self.default_dimension}d")
        self.assertIsInstance(self.embedder, TextEmbedder)

    def test_initialization_with_custom_dimension(self):
        """Given: 커스텀 차원으로 초기화
        When: 어댑터를 생성하면
        Then: 올바른 속성값들이 설정된다
        """
        # Given & When: setUp에서 수행됨

        # Then
        self.assertEqual(self.custom_embedder._dimension, self.custom_dimension)
        self.assertEqual(self.custom_embedder.model_name, f"random-{self.custom_dimension}d")

    # get_embedding_dimension 테스트
    def test_get_embedding_dimension(self):
        """Given: 초기화된 어댑터
        When: get_embedding_dimension을 호출하면
        Then: 설정된 차원을 반환한다
        """
        # Given: setUp에서 수행됨

        # When
        dimension = self.embedder.get_embedding_dimension()
        custom_dimension = self.custom_embedder.get_embedding_dimension()

        # Then
        self.assertEqual(dimension, self.default_dimension)
        self.assertEqual(custom_dimension, self.custom_dimension)

    # is_available 테스트
    async def test_is_available(self):
        """Given: 초기화된 어댑터
        When: is_available을 호출하면
        Then: 항상 True를 반환한다
        """
        # Given: setUp에서 수행됨

        # When
        is_available = await self.embedder.is_available()

        # Then
        self.assertTrue(is_available)

    # embed_text 성공 케이스 테스트
    async def test_embed_text_success(self):
        """Given: 유효한 텍스트
        When: embed_text를 호출하면
        Then: 올바른 EmbeddingResult를 반환한다
        """
        # Given: setUp에서 test_text 설정됨

        # When
        result = await self.embedder.embed_text(self.test_text)

        # Then
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.text, self.test_text)
        self.assertEqual(result.model_name, f"random-{self.default_dimension}d")
        self.assertEqual(result.dimension, self.default_dimension)
        self.assertEqual(len(result.embedding), self.default_dimension)
        self.assertIsInstance(result.processing_time_ms, float)
        self.assertGreaterEqual(result.processing_time_ms, 0)

        # 메타데이터 검증
        self.assertIn("seed", result.metadata)
        self.assertIn("normalized", result.metadata)
        self.assertTrue(result.metadata["normalized"])

        # 벡터 정규화 검증 (L2 norm이 1에 가까워야 함)
        embedding_array = np.array(result.embedding)
        norm = np.linalg.norm(embedding_array)
        self.assertAlmostEqual(norm, 1.0, places=6)

    async def test_embed_text_deterministic(self):
        """Given: 동일한 텍스트
        When: 여러 번 embed_text를 호출하면
        Then: 항상 같은 벡터를 반환한다
        """
        # Given: setUp에서 test_text 설정됨

        # When
        result1 = await self.embedder.embed_text(self.test_text)
        result2 = await self.embedder.embed_text(self.test_text)

        # Then
        self.assertEqual(result1.embedding, result2.embedding)
        self.assertEqual(result1.metadata["seed"], result2.metadata["seed"])

    async def test_embed_text_different_inputs_different_outputs(self):
        """Given: 서로 다른 텍스트들
        When: embed_text를 호출하면
        Then: 서로 다른 벡터를 반환한다
        """
        # Given
        text1 = "첫 번째 텍스트"
        text2 = "두 번째 텍스트"

        # When
        result1 = await self.embedder.embed_text(text1)
        result2 = await self.embedder.embed_text(text2)

        # Then
        self.assertNotEqual(result1.embedding, result2.embedding)
        self.assertNotEqual(result1.metadata["seed"], result2.metadata["seed"])

    # embed_texts 테스트
    async def test_embed_texts_success(self):
        """Given: 유효한 텍스트 리스트
        When: embed_texts를 호출하면
        Then: 모든 텍스트에 대한 EmbeddingResult 리스트를 반환한다
        """
        # Given
        texts = ["첫 번째 텍스트", "두 번째 텍스트", "세 번째 텍스트"]

        # When
        results = await self.embedder.embed_texts(texts)

        # Then
        self.assertEqual(len(results), len(texts))
        for i, result in enumerate(results):
            self.assertIsInstance(result, EmbeddingResult)
            self.assertEqual(result.text, texts[i])
            self.assertEqual(result.dimension, self.default_dimension)
            self.assertEqual(len(result.embedding), self.default_dimension)

    async def test_embed_texts_empty_list(self):
        """Given: 빈 텍스트 리스트
        When: embed_texts를 호출하면
        Then: 빈 리스트를 반환한다
        """
        # Given
        texts = []

        # When
        results = await self.embedder.embed_texts(texts)

        # Then
        self.assertEqual(results, [])

    async def test_embed_texts_consistency_with_embed_text(self):
        """Given: 동일한 텍스트
        When: embed_text와 embed_texts를 각각 호출하면
        Then: 같은 결과를 반환한다
        """
        # Given
        test_text = "일관성 테스트 텍스트"

        # When
        single_result = await self.embedder.embed_text(test_text)
        batch_results = await self.embedder.embed_texts([test_text])

        # Then
        self.assertEqual(len(batch_results), 1)
        batch_result = batch_results[0]

        self.assertEqual(single_result.text, batch_result.text)
        self.assertEqual(single_result.embedding, batch_result.embedding)
        self.assertEqual(single_result.metadata["seed"], batch_result.metadata["seed"])

    # _text_to_seed 내부 메서드 테스트
    def test_text_to_seed_deterministic(self):
        """Given: 동일한 텍스트
        When: _text_to_seed를 호출하면
        Then: 항상 같은 시드를 반환한다
        """
        # Given: setUp에서 test_text 설정됨

        # When
        seed1 = self.embedder._text_to_seed(self.test_text)
        seed2 = self.embedder._text_to_seed(self.test_text)

        # Then
        self.assertEqual(seed1, seed2)

    def test_text_to_seed_different_inputs(self):
        """Given: 서로 다른 텍스트들
        When: _text_to_seed를 호출하면
        Then: 서로 다른 시드를 반환한다
        """
        # Given
        text1 = "첫 번째 텍스트"
        text2 = "두 번째 텍스트"

        # When
        seed1 = self.embedder._text_to_seed(text1)
        seed2 = self.embedder._text_to_seed(text2)

        # Then
        self.assertNotEqual(seed1, seed2)

    def test_text_to_seed_range(self):
        """Given: 임의의 텍스트
        When: _text_to_seed를 호출하면
        Then: 유효한 범위의 시드를 반환한다
        """
        # Given: setUp에서 test_text 설정됨

        # When
        seed = self.embedder._text_to_seed(self.test_text)

        # Then
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    # 성능 관련 테스트
    async def test_embed_texts_parallel_processing(self):
        """Given: 여러 텍스트
        When: embed_texts를 호출하면
        Then: 병렬 처리로 효율적으로 실행된다
        """
        # Given
        texts = [f"텍스트 {i}" for i in range(5)]

        # When
        start_time = asyncio.get_event_loop().time()
        results = await self.embedder.embed_texts(texts)
        end_time = asyncio.get_event_loop().time()

        # Then
        self.assertEqual(len(results), len(texts))
        # 병렬 처리이므로 순차 처리보다 빨라야 함 (단, 매우 빠른 연산이므로 절대 시간보다는 로직 검증에 집중)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)  # 1초 이내 완료

    # 다양한 차원 테스트
    async def test_different_dimensions(self):
        """Given: 다양한 차원의 어댑터들
        When: embed_text를 호출하면
        Then: 각각 설정된 차원의 벡터를 반환한다
        """
        # Given
        dimensions = [32, 64, 128, 256, 512]

        for dimension in dimensions:
            with self.subTest(dimension=dimension):
                # When
                embedder = RandomTextEmbedder(dimension)
                result = await embedder.embed_text(self.test_text)

                # Then
                self.assertEqual(result.dimension, dimension)
                self.assertEqual(len(result.embedding), dimension)

                # 정규화 검증
                embedding_array = np.array(result.embedding)
                norm = np.linalg.norm(embedding_array)
                self.assertAlmostEqual(norm, 1.0, places=6)

    # 모킹을 통한 시드 고정 테스트
    @patch("numpy.random.seed")
    @patch("numpy.random.randn")
    async def test_embed_text_with_mocked_numpy(self, mock_randn, mock_seed):
        """Given: 모킹된 numpy 함수들
        When: embed_text를 호출하면
        Then: 예상된 시드와 벡터 생성 과정을 거친다
        """
        # Given
        mock_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_randn.return_value = mock_vector

        # When
        embedder = RandomTextEmbedder(3)
        result = await embedder.embed_text(self.test_text)

        # Then
        expected_seed = embedder._text_to_seed(self.test_text)
        mock_seed.assert_called_once_with(expected_seed)
        mock_randn.assert_called_once_with(3)

        # 정규화된 벡터 검증
        expected_normalized = mock_vector / np.linalg.norm(mock_vector)
        self.assertEqual(result.embedding, expected_normalized.tolist())

    # 동시성 테스트
    async def test_concurrent_embed_text_calls(self):
        """Given: 동시에 여러 embed_text 호출
        When: 병렬로 실행하면
        Then: 각각 올바른 결과를 반환한다
        """
        # Given
        texts = [f"동시성 테스트 {i}" for i in range(10)]

        # When
        tasks = [self.embedder.embed_text(text) for text in texts]
        results = await asyncio.gather(*tasks)

        # Then
        self.assertEqual(len(results), len(texts))
        for i, result in enumerate(results):
            self.assertEqual(result.text, texts[i])
            self.assertEqual(result.dimension, self.default_dimension)

        # 모든 결과가 서로 다른지 확인
        embeddings = [result.embedding for result in results]
        for i, embedding_i in enumerate(embeddings):
            for embedding_j in embeddings[i + 1 :]:
                self.assertNotEqual(embedding_i, embedding_j)


if __name__ == "__main__":
    unittest.main()
