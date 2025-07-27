"""
OpenAI TextEmbedder 어댑터 단위 테스트.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.adapters.openai.text_embedder import OpenAITextEmbedder
from src.domain.value_objects.vector import Vector


class TestOpenAITextEmbedder(unittest.TestCase):
    """OpenAI TextEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # 환경 변수 모킹
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            with patch('src.adapters.openai.text_embedder.openai'):
                self.embedder = OpenAITextEmbedder(api_key="test-key")
                
    def test_initialization(self):
        """초기화 테스트."""
        self.assertEqual(self.embedder.api_key, "test-key")
        self.assertEqual(self.embedder.model, "text-embedding-3-small")
        self.assertEqual(self.embedder._dimension, 1536)
        
    def test_initialization_without_api_key_raises_error(self):
        """API 키 없이 초기화 시 에러 발생 테스트."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('src.adapters.openai.text_embedder.openai'):
                with self.assertRaises(ValueError):
                    OpenAITextEmbedder()
                    
    def test_get_embedding_dimension(self):
        """임베딩 차원 반환 테스트."""
        dimension = self.embedder.get_embedding_dimension()
        self.assertEqual(dimension, 1536)
        
    def test_get_model_name(self):
        """모델명 반환 테스트."""
        model_name = self.embedder.get_model_name()
        self.assertEqual(model_name, "text-embedding-3-small")
        
    def test_get_max_token_length(self):
        """최대 토큰 길이 반환 테스트."""
        max_tokens = self.embedder.get_max_token_length()
        self.assertEqual(max_tokens, 8192)
        
    async def test_embed_text(self):
        """텍스트 임베딩 테스트."""
        # Mock OpenAI 클라이언트 응답
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        self.embedder.client.embeddings.create = Mock(return_value=mock_response)
        
        result = await self.embedder.embed_text("테스트 텍스트")
        
        self.assertIsInstance(result, Vector)
        self.assertEqual(result.values, [0.1, 0.2, 0.3])
        
    async def test_embed_texts(self):
        """여러 텍스트 임베딩 테스트."""
        # Mock OpenAI 클라이언트 응답
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        self.embedder.client.embeddings.create = Mock(return_value=mock_response)
        
        result = await self.embedder.embed_texts(["텍스트1", "텍스트2"])
        
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Vector)
        self.assertIsInstance(result[1], Vector)
        self.assertEqual(result[0].values, [0.1, 0.2, 0.3])
        self.assertEqual(result[1].values, [0.4, 0.5, 0.6])
        
    async def test_compute_similarity(self):
        """벡터 유사도 계산 테스트."""
        vector1 = Vector([1.0, 0.0, 0.0])
        vector2 = Vector([0.0, 1.0, 0.0])
        
        similarity = await self.embedder.compute_similarity(vector1, vector2)
        
        self.assertAlmostEqual(similarity, 0.0, places=5)  # 직교 벡터는 유사도 0
        
    def test_preprocess_text(self):
        """텍스트 전처리 테스트."""
        text = "  여러개의    공백이   있는   텍스트  "
        processed = self.embedder.preprocess_text(text)
        
        self.assertEqual(processed, "여러개의 공백이 있는 텍스트")
        
    async def test_validate_embedding(self):
        """임베딩 벡터 검증 테스트."""
        # 유효한 벡터
        valid_vector = Vector([0.1] * 1536)
        self.assertTrue(await self.embedder.validate_embedding(valid_vector))
        
        # 잘못된 차원의 벡터
        invalid_vector = Vector([0.1] * 100)
        self.assertFalse(await self.embedder.validate_embedding(invalid_vector))
        
        # NaN이 포함된 벡터
        nan_vector = Vector([float('nan')] * 1536)
        self.assertFalse(await self.embedder.validate_embedding(nan_vector))


if __name__ == "__main__":
    unittest.main()