"""
OpenAI TextEmbedder 어댑터 단위 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from src.adapters.openai.text_embedder import OpenAITextEmbedder
from src.domain.value_objects.vector import Vector


class TestOpenAITextEmbedder(unittest.IsolatedAsyncioTestCase):
    """OpenAI TextEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # 환경 변수 모킹
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}):
            with patch("src.adapters.openai.text_embedder.openai"):
                self.embedder = OpenAITextEmbedder(api_key="test-key")

    def test_initialization(self):
        """초기화 테스트."""
        self.assertEqual(self.embedder.api_key, "test-key")
        self.assertEqual(self.embedder.model, "text-embedding-3-small")
        self.assertEqual(self.embedder._dimension, 1536)

    def test_initialization_without_api_key_raises_error(self):
        """API 키 없이 초기화 시 에러 발생 테스트."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.adapters.openai.text_embedder.openai"):
                with self.assertRaises(ValueError):
                    OpenAITextEmbedder()

    def test_get_embedding_dimension(self):
        """임베딩 차원 반환 테스트."""
        dimension = self.embedder.get_embedding_dimension()
        self.assertEqual(dimension, 1536)


    async def test_embed_text(self):
        """텍스트 임베딩 테스트."""
        # Mock OpenAI 클라이언트 응답
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

        self.embedder.client.embeddings.create = Mock(return_value=mock_response)

        result = await self.embedder.embed_text("테스트 텍스트")

        self.assertIsInstance(result, Vector)
        self.assertEqual(len(result.values), 3)
        self.assertAlmostEqual(result.values[0], 0.1, places=5)
        self.assertAlmostEqual(result.values[1], 0.2, places=5)
        self.assertAlmostEqual(result.values[2], 0.3, places=5)

    async def test_embed_texts(self):
        """여러 텍스트 임베딩 테스트."""
        # Mock OpenAI 클라이언트 응답
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]

        self.embedder.client.embeddings.create = Mock(return_value=mock_response)

        result = await self.embedder.embed_texts(["텍스트1", "텍스트2"])

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Vector)
        self.assertIsInstance(result[1], Vector)
        # 첫 번째 벡터 검증
        self.assertEqual(len(result[0].values), 3)
        self.assertAlmostEqual(result[0].values[0], 0.1, places=5)
        self.assertAlmostEqual(result[0].values[1], 0.2, places=5)
        self.assertAlmostEqual(result[0].values[2], 0.3, places=5)
        
        # 두 번째 벡터 검증
        self.assertEqual(len(result[1].values), 3)
        self.assertAlmostEqual(result[1].values[0], 0.4, places=5)
        self.assertAlmostEqual(result[1].values[1], 0.5, places=5)
        self.assertAlmostEqual(result[1].values[2], 0.6, places=5)

    async def test_is_available(self):
        """서비스 가용성 확인 테스트."""
        # Mock OpenAI 클라이언트 응답 
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        self.embedder.client.embeddings.create = Mock(return_value=mock_response)

        result = await self.embedder.is_available()
        self.assertTrue(result)

        # 실패 케이스
        self.embedder.client.embeddings.create = Mock(side_effect=Exception("API Error"))
        result = await self.embedder.is_available()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
