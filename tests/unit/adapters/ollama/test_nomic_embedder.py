"""
Nomic Embedder 어댑터 단위 테스트.
"""

# pylint: disable=protected-access
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from src.adapters.ollama.nomic_embedder import NomicEmbedder, create_nomic_embedder


class TestNomicEmbedder(unittest.TestCase):
    """NomicEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # Mock requests.Session to avoid actual HTTP calls
        with patch("src.adapters.ollama.nomic_embedder.requests.Session") as mock_session_cls:
            self.mock_session = Mock()
            mock_session_cls.return_value = self.mock_session

            # Mock successful model availability check
            mock_tags_response = Mock()
            mock_tags_response.status_code = 200
            mock_tags_response.json.return_value = {"models": [{"name": "nomic-embed-text"}]}
            self.mock_session.get.return_value = mock_tags_response

            self.embedder = NomicEmbedder(
                base_url="http://localhost:11434", model_name="nomic-embed-text", timeout=60
            )

    def test_initialization_default_params(self):
        """기본 파라미터로 초기화 테스트."""
        # Given: Mock session and successful model check
        with patch("src.adapters.ollama.nomic_embedder.requests.Session") as mock_session_cls:
            mock_session = Mock()
            mock_session_cls.return_value = mock_session

            mock_tags_response = Mock()
            mock_tags_response.status_code = 200
            mock_tags_response.json.return_value = {"models": [{"name": "nomic-embed-text"}]}
            mock_session.get.return_value = mock_tags_response

            # When: Create embedder with defaults
            embedder = NomicEmbedder()

            # Then: Default values should be set
            self.assertEqual(embedder.base_url, "http://localhost:11434")
            self.assertEqual(embedder.model_name, "nomic-embed-text")
            self.assertEqual(embedder.timeout, 60)

    def test_initialization_custom_params(self):
        """커스텀 파라미터로 초기화 테스트."""
        self.assertEqual(self.embedder.base_url, "http://localhost:11434")
        self.assertEqual(self.embedder.model_name, "nomic-embed-text")
        self.assertEqual(self.embedder.timeout, 60)

    def test_ensure_model_available_model_exists(self):
        """모델이 이미 존재하는 경우 테스트."""
        # Given: Model is already available
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "nomic-embed-text"}]}
        self.mock_session.get.return_value = mock_response

        # When: Check model availability
        result = self.embedder._ensure_model_available()

        # Then: Should return True
        self.assertTrue(result)
        self.mock_session.get.assert_called_with("http://localhost:11434/api/tags", timeout=10)

    def test_ensure_model_available_model_not_exists_pull_success(self):
        """모델이 없어서 풀링에 성공하는 경우 테스트."""
        # Given: Model not available initially, but pull succeeds
        mock_tags_response = Mock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = Mock()
        mock_pull_response.status_code = 200

        self.mock_session.get.return_value = mock_tags_response
        self.mock_session.post.return_value = mock_pull_response

        # When: Check model availability
        result = self.embedder._ensure_model_available()

        # Then: Should return True after successful pull
        self.assertTrue(result)
        self.mock_session.post.assert_called_with(
            "http://localhost:11434/api/pull", json={"name": "nomic-embed-text"}, timeout=300
        )

    def test_ensure_model_available_connection_error(self):
        """연결 오류 발생 시 테스트."""
        # Given: Connection error occurs
        self.mock_session.get.side_effect = requests.ConnectionError("Connection failed")

        # When: Check model availability
        result = self.embedder._ensure_model_available()

        # Then: Should return False
        self.assertFalse(result)

    def test_embed_single_text_success(self):
        """단일 텍스트 임베딩 성공 테스트."""
        # Given: Successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        self.mock_session.post.return_value = mock_response

        # When: Embed single text
        result = self.embedder.embed("test text")

        # Then: Should return correct embedding
        expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        self.mock_session.post.assert_called_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test text"},
            timeout=60,
        )

    def test_embed_multiple_texts_success(self):
        """여러 텍스트 임베딩 성공 테스트."""
        # Given: Successful API responses
        responses = [
            Mock(status_code=200, json=lambda: {"embedding": [0.1, 0.2]}),
            Mock(status_code=200, json=lambda: {"embedding": [0.3, 0.4]}),
        ]
        self.mock_session.post.side_effect = responses

        # When: Embed multiple texts
        result = self.embedder.embed(["text1", "text2"])

        # Then: Should return array of embeddings
        expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_embed_connection_error(self):
        """연결 오류 시 제로 벡터 반환 테스트."""
        # Given: Connection error occurs
        self.mock_session.post.side_effect = requests.ConnectionError("Connection failed")

        # When: Embed text
        result = self.embedder.embed("test text")

        # Then: Should return zero vector with default dimension
        expected = np.zeros(768, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_embed_http_error(self):
        """HTTP 오류 시 제로 벡터 반환 테스트."""
        # Given: HTTP error occurs
        self.mock_session.post.side_effect = requests.HTTPError("HTTP 500 Error")

        # When: Embed text
        result = self.embedder.embed("test text")

        # Then: Should return zero vector with default dimension
        expected = np.zeros(768, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_embed_json_decode_error(self):
        """JSON 파싱 오류 시 제로 벡터 반환 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        self.mock_session.post.return_value = mock_response

        # When: Embed text
        result = self.embedder.embed("test text")

        # Then: Should return zero vector with default dimension
        expected = np.zeros(768, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_embed_batch(self):
        """배치 임베딩 테스트."""
        # Given: Successful embedding responses
        mock_responses = [
            Mock(status_code=200, json=lambda: {"embedding": [0.1, 0.2]}),
            Mock(status_code=200, json=lambda: {"embedding": [0.3, 0.4]}),
        ]
        self.mock_session.post.side_effect = mock_responses

        # When: Embed batch of texts
        result = self.embedder.embed_batch(["text1", "text2"])

        # Then: Should return list of embeddings
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], np.array([0.1, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.array([0.3, 0.4], dtype=np.float32))

    def test_dimension_property_success(self):
        """임베딩 차원 속성 성공 테스트."""
        # Given: Successful test embedding
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        self.mock_session.post.return_value = mock_response

        # When: Get dimension
        dimension = self.embedder.dimension

        # Then: Should return actual embedding dimension
        self.assertEqual(dimension, 4)

    def test_dimension_property_error_fallback(self):
        """임베딩 차원 속성 오류 시 기본값 반환 테스트."""
        # Given: Connection error during test embedding
        self.mock_session.post.side_effect = requests.ConnectionError("Connection failed")

        # When: Get dimension
        dimension = self.embedder.dimension

        # Then: Should return default dimension
        self.assertEqual(dimension, 768)

    def test_get_embedding_dimension(self):
        """임베딩 차원 getter 메서드 테스트."""
        # Given: Mock successful embedding response for dimension detection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}  # 5-dimensional
        self.mock_session.post.return_value = mock_response

        # When: Get embedding dimension
        dimension = self.embedder.get_embedding_dimension()

        # Then: Should return actual dimension
        self.assertEqual(dimension, 5)

    def test_batch_embed_single_batch(self):
        """단일 배치 임베딩 테스트."""
        # Given: Successful embedding responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        self.mock_session.post.return_value = mock_response

        # When: Batch embed texts
        result = self.embedder.batch_embed(["text1"], batch_size=32)

        # Then: Should return array with single embedding
        # Debug: Print actual result shape and values
        print(f"Actual result shape: {result.shape}")
        print(f"Actual result: {result}")

        # After fixing batch_embed, it returns (1, 2) shape for single text
        expected_values = np.array([[0.1, 0.2]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected_values)

    def test_batch_embed_multiple_batches(self):
        """다중 배치 임베딩 테스트."""
        # Given: Mock multiple embedding responses
        # batch_size=2 means:
        # - First batch: ["text1", "text2"] -> embed() called twice for individual texts
        # - Second batch: ["text3"] -> embed() called once for single text

        def mock_post_side_effect(url, json_data=None, timeout=None):
            prompt = json_data["prompt"]
            if prompt == "text1":
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"embedding": [0.1, 0.2]}
                return response
            if prompt == "text2":
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"embedding": [0.3, 0.4]}
                return response
            if prompt == "text3":
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"embedding": [0.5, 0.6]}
                return response
            else:
                raise ValueError(f"Unexpected prompt: {prompt}")

        self.mock_session.post.side_effect = mock_post_side_effect

        # When: Batch embed with small batch size
        result = self.embedder.batch_embed(["text1", "text2", "text3"], batch_size=2)

        # Then: Should return all embeddings as a 2D array
        # Each embedding is a 1D array, collected into a 2D array
        expected = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_almost_equal(result, expected)

    def test_similarity_calculation(self):
        """코사인 유사도 계산 테스트."""
        # Given: Two embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])

        # When: Calculate similarity
        similarity = self.embedder.similarity(embedding1, embedding2)

        # Then: Should return cosine similarity (0 for perpendicular vectors)
        self.assertAlmostEqual(similarity, 0.0, places=5)

    def test_similarity_identical_vectors(self):
        """동일한 벡터 간 유사도 테스트."""
        # Given: Identical embeddings
        embedding1 = np.array([1.0, 1.0, 1.0])
        embedding2 = np.array([1.0, 1.0, 1.0])

        # When: Calculate similarity
        similarity = self.embedder.similarity(embedding1, embedding2)

        # Then: Should return 1.0 for identical vectors
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_similarity_zero_vectors(self):
        """제로 벡터 간 유사도 테스트."""
        # Given: Zero embeddings
        embedding1 = np.array([0.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 0.0, 0.0])

        # When: Calculate similarity
        similarity = self.embedder.similarity(embedding1, embedding2)

        # Then: Should return 0.0 for zero vectors
        self.assertEqual(similarity, 0.0)

    def test_most_similar_texts(self):
        """가장 유사한 텍스트 찾기 테스트."""
        # Given: Mock embedding responses
        responses = [
            Mock(status_code=200, json=lambda: {"embedding": [1.0, 0.0]}),  # Query
            Mock(status_code=200, json=lambda: {"embedding": [0.9, 0.1]}),  # Similar
            Mock(status_code=200, json=lambda: {"embedding": [0.1, 0.9]}),  # Different
        ]
        self.mock_session.post.side_effect = responses

        # When: Find most similar texts
        result = self.embedder.most_similar_texts(
            "query text", ["similar text", "different text"], top_k=2
        )

        # Then: Should return ranked results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "similar text")  # Most similar first
        self.assertEqual(result[1][0], "different text")
        self.assertGreater(result[0][1], result[1][1])  # Higher similarity score


class TestCreateNomicEmbedder(unittest.TestCase):
    """create_nomic_embedder 팩토리 함수 테스트."""

    def test_create_nomic_embedder_default_params(self):
        """기본 파라미터로 팩토리 함수 테스트."""
        # Given: Mock session to avoid actual HTTP calls
        with patch("src.adapters.ollama.nomic_embedder.requests.Session") as mock_session_cls:
            mock_session = Mock()
            mock_session_cls.return_value = mock_session

            # Mock successful model check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "nomic-embed-text"}]}
            mock_session.get.return_value = mock_response

            # When: Create embedder using factory
            embedder = create_nomic_embedder()

            # Then: Should return configured embedder
            self.assertIsInstance(embedder, NomicEmbedder)
            self.assertEqual(embedder.base_url, "http://localhost:11434")
            self.assertEqual(embedder.model_name, "nomic-embed-text")

    def test_create_nomic_embedder_custom_params(self):
        """커스텀 파라미터로 팩토리 함수 테스트."""
        # Given: Mock session to avoid actual HTTP calls
        with patch("src.adapters.ollama.nomic_embedder.requests.Session") as mock_session_cls:
            mock_session = Mock()
            mock_session_cls.return_value = mock_session

            # Mock successful model check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "custom-model"}]}
            mock_session.get.return_value = mock_response

            # When: Create embedder with custom parameters
            embedder = create_nomic_embedder(
                base_url="http://custom:8080", model_name="custom-model"
            )

            # Then: Should return configured embedder with custom settings
            self.assertIsInstance(embedder, NomicEmbedder)
            self.assertEqual(embedder.base_url, "http://custom:8080")
            self.assertEqual(embedder.model_name, "custom-model")


if __name__ == "__main__":
    unittest.main()
