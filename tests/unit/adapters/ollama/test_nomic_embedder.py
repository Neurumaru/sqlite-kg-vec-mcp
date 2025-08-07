"""
Nomic Embedder 어댑터 단위 테스트.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from src.adapters.ollama.exceptions import OllamaModelException
from src.adapters.ollama.nomic_embedder import NomicEmbedder
from src.common.config.llm import OllamaConfig
from src.dto.embedding import EmbeddingResult


class TestNomicEmbedder(unittest.TestCase):
    """NomicEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.mock_client = AsyncMock()
        self.model_name = "nomic-embed-text"
        self.config = OllamaConfig(embedding_dimension=768)
        self.embedder = NomicEmbedder(
            client=self.mock_client, model_name=self.model_name, config=self.config
        )

    def test_initialization_with_config(self):
        """설정과 함께 초기화 테스트."""
        # Given & When: 설정과 함께 임베더 생성
        config = OllamaConfig(embedding_dimension=512)
        embedder = NomicEmbedder(client=self.mock_client, model_name="custom-model", config=config)

        # Then: 설정이 올바르게 적용되어야 함
        self.assertEqual(embedder.model_name, "custom-model")
        self.assertEqual(embedder.dimension, 512)
        self.assertEqual(embedder.client, self.mock_client)

    def test_initialization_without_config(self):
        """설정 없이 초기화 테스트."""
        # Given & When: 설정 없이 임베더 생성
        embedder = NomicEmbedder(client=self.mock_client, model_name="test-model")

        # Then: 기본 설정이 적용되어야 함
        self.assertEqual(embedder.model_name, "test-model")
        self.assertEqual(embedder.dimension, 768)  # 기본값
        self.assertEqual(embedder.client, self.mock_client)

    def test_embed_text_success(self):
        """텍스트 임베딩 성공 테스트."""
        # Given: 성공적인 API 응답
        mock_response = {"embedding": [0.1, 0.2, 0.3] * 256}  # 768차원
        self.mock_client.embed.return_value = mock_response

        # When: 텍스트 임베딩 실행
        result = asyncio.run(self.embedder.embed_text("test text"))

        # Then: 올바른 결과가 반환되어야 함
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.text, "test text")
        self.assertEqual(result.embedding, mock_response["embedding"])
        self.assertEqual(result.model_name, self.model_name)
        self.assertEqual(result.dimension, 768)

        # API가 올바른 파라미터로 호출되었는지 확인
        self.mock_client.embed.assert_called_once_with(model=self.model_name, input="test text")

    def test_embed_text_dimension_mismatch(self):
        """차원 불일치 시 예외 발생 테스트."""
        # Given: 잘못된 차원의 임베딩 응답
        mock_response = {"embedding": [0.1, 0.2]}  # 2차원 (예상: 768차원)
        self.mock_client.embed.return_value = mock_response

        # When & Then: 예외가 발생해야 함
        with self.assertRaises(OllamaModelException) as context:
            asyncio.run(self.embedder.embed_text("test text"))

        self.assertIn("예상 임베딩 차원", str(context.exception))
        self.assertIn("768", str(context.exception))
        self.assertIn("2", str(context.exception))

    def test_embed_text_api_error(self):
        """API 오류 시 예외 발생 테스트."""
        # Given: API 호출에서 예외 발생
        self.mock_client.embed.side_effect = ConnectionError("API Error")

        # When & Then: OllamaModelException이 발생해야 함
        with self.assertRaises(OllamaModelException) as context:
            asyncio.run(self.embedder.embed_text("test text"))

        self.assertIn("텍스트 임베딩 실패", str(context.exception))
        self.assertIn("API Error", str(context.exception))

    def test_embed_texts_success(self):
        """여러 텍스트 임베딩 성공 테스트."""
        # Given: 성공적인 API 응답들
        embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        self.mock_client.embed.side_effect = [
            {"embedding": embeddings[0]},
            {"embedding": embeddings[1]},
            {"embedding": embeddings[2]},
        ]

        texts = ["text1", "text2", "text3"]

        # When: 여러 텍스트 임베딩 실행
        results = asyncio.run(self.embedder.embed_texts(texts))

        # Then: 올바른 결과들이 반환되어야 함
        self.assertEqual(len(results), 3)

        for i, result in enumerate(results):
            self.assertIsInstance(result, EmbeddingResult)
            self.assertEqual(result.text, texts[i])
            self.assertEqual(result.embedding, embeddings[i])
            self.assertEqual(result.model_name, self.model_name)
            self.assertEqual(result.dimension, 768)

        # API가 각 텍스트에 대해 호출되었는지 확인
        self.assertEqual(self.mock_client.embed.call_count, 3)

    def test_embed_texts_empty_list(self):
        """빈 텍스트 목록 임베딩 테스트."""
        # Given: 빈 텍스트 목록
        texts = []

        # When: 빈 목록 임베딩 실행
        results = asyncio.run(self.embedder.embed_texts(texts))

        # Then: 빈 결과 목록이 반환되어야 함
        self.assertEqual(len(results), 0)
        self.mock_client.embed.assert_not_called()

    def test_is_available_success(self):
        """서비스 가용성 확인 성공 테스트."""
        # Given: 성공적인 list 호출
        self.mock_client.list.return_value = Mock()

        # When: 가용성 확인
        result = asyncio.run(self.embedder.is_available())

        # Then: True가 반환되어야 함
        self.assertTrue(result)
        self.mock_client.list.assert_called_once()

    def test_is_available_failure(self):
        """서비스 가용성 확인 실패 테스트."""
        # Given: list 호출에서 예외 발생
        self.mock_client.list.side_effect = Exception("Connection error")

        # When: 가용성 확인
        result = asyncio.run(self.embedder.is_available())

        # Then: False가 반환되어야 함
        self.assertFalse(result)
        self.mock_client.list.assert_called_once()

    def test_embed_texts_partial_failure(self):
        """일부 텍스트 임베딩 실패 테스트."""
        # Given: 두 번째 호출에서 예외 발생
        self.mock_client.embed.side_effect = [
            {"embedding": [0.1] * 768},  # 성공
            Exception("API Error"),  # 실패
            {"embedding": [0.3] * 768},  # 성공
        ]

        texts = ["text1", "text2", "text3"]

        # When & Then: 두 번째 텍스트에서 예외가 발생해야 함
        with self.assertRaises(OllamaModelException):
            asyncio.run(self.embedder.embed_texts(texts))


if __name__ == "__main__":
    unittest.main()
