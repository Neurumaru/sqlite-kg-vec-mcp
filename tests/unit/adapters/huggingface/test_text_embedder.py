"""
HuggingFace 텍스트 임베딩 어댑터 테스트.
"""

# pylint: disable=protected-access

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.huggingface.exceptions import (
    HuggingFaceEmbeddingException,
    HuggingFaceModelLoadException,
)
from src.adapters.huggingface.text_embedder import HuggingFaceTextEmbedder
from src.dto import EmbeddingResult


class TestHuggingFaceTextEmbedder(unittest.IsolatedAsyncioTestCase):
    """HuggingFaceTextEmbedder 단위 테스트."""

    def setUp(self):
        """테스트 설정."""
        # Mock SentenceTransformer
        self.mock_model = Mock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        # 384차원에 맞는 임베딩 벡터 생성
        self.mock_embedding_384d = np.random.rand(384).astype(np.float32)
        self.mock_model.encode.return_value = self.mock_embedding_384d

        # Mock 로거
        self.mock_logger = Mock()

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    def test_initialization_success(self, mock_get_logger, mock_sentence_transformer):
        """Given: 유효한 모델명과 설치된 sentence-transformers
        When: HuggingFaceTextEmbedder를 초기화할 때
        Then: 성공적으로 초기화되어야 함"""
        # Given
        model_name = "all-MiniLM-L6-v2"
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger

        # When
        embedder = HuggingFaceTextEmbedder(model_name)

        # Then
        self.assertEqual(embedder.model_name, model_name)
        self.assertEqual(embedder._dimension, 384)
        mock_sentence_transformer.assert_called_once_with(model_name)
        self.mock_logger.info.assert_called_once()

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    def test_initialization_import_error(self):
        """Given: sentence-transformers가 설치되지 않은 상태
        When: HuggingFaceTextEmbedder를 초기화할 때
        Then: ImportError가 발생해야 함"""
        # When & Then
        with self.assertRaises(ImportError) as context:
            HuggingFaceTextEmbedder()

        self.assertIn("sentence-transformers가 설치되지 않았습니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    def test_initialization_empty_model_name(self):
        """Given: 빈 모델명
        When: HuggingFaceTextEmbedder를 초기화할 때
        Then: ValueError가 발생해야 함"""
        # When & Then
        with self.assertRaises(ValueError) as context:
            HuggingFaceTextEmbedder("")

        self.assertIn("모델명은 비어있을 수 없습니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    def test_initialization_model_load_error(self, mock_get_logger, mock_sentence_transformer):
        """Given: SentenceTransformer 로딩 중 오류 발생
        When: HuggingFaceTextEmbedder를 초기화할 때
        Then: HuggingFaceModelLoadException이 발생해야 함"""
        # Given
        original_error = Exception("Model loading failed")
        mock_sentence_transformer.side_effect = original_error
        mock_get_logger.return_value = self.mock_logger

        # When & Then
        with self.assertRaises(HuggingFaceModelLoadException) as context:
            HuggingFaceTextEmbedder("invalid-model")

        # 예외 내용 검증
        exception = context.exception
        self.assertEqual(exception.model_name, "invalid-model")
        self.assertEqual(exception.operation, "model loading")
        self.assertEqual(exception.error_code, "HUGGINGFACE_MODEL_LOAD_FAILED")
        self.assertEqual(exception.original_error, original_error)
        self.assertIn("모델 로딩 실패", str(exception))

        self.mock_logger.error.assert_called_once()

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    def test_initialization_dimension_none(self, mock_get_logger, mock_sentence_transformer):
        """Given: 모델의 embedding dimension이 None인 경우
        When: HuggingFaceTextEmbedder를 초기화할 때
        Then: HuggingFaceModelLoadException이 발생해야 함"""
        # Given
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = None
        mock_sentence_transformer.return_value = mock_model
        mock_get_logger.return_value = self.mock_logger

        # When & Then
        with self.assertRaises(HuggingFaceModelLoadException) as context:
            HuggingFaceTextEmbedder("test-model")

        # 예외 내용 검증
        exception = context.exception
        self.assertEqual(exception.model_name, "test-model")
        self.assertEqual(exception.operation, "model loading")
        self.assertEqual(exception.error_code, "HUGGINGFACE_MODEL_LOAD_FAILED")
        self.assertIn("Unable to determine embedding dimension", str(exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_text_success(self, mock_get_logger, mock_sentence_transformer):
        """Given: 정상적으로 초기화된 임베더와 유효한 텍스트
        When: embed_text를 호출할 때
        Then: EmbeddingResult가 반환되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        text = "Hello, world!"
        expected_embedding = np.random.rand(384).astype(np.float32)
        self.mock_model.encode.return_value = expected_embedding

        # When
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.001]  # 시작 시간, 종료 시간
            result = await embedder.embed_text(text)

        # Then
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.text, text)
        self.assertEqual(result.embedding, expected_embedding.tolist())
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.dimension, 384)
        self.assertEqual(result.processing_time_ms, 1.0)
        self.mock_model.encode.assert_called_once_with(
            text, convert_to_numpy=True, show_progress_bar=False
        )

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_text_empty_text(self, mock_get_logger, mock_sentence_transformer):
        """Given: 빈 텍스트
        When: embed_text를 호출할 때
        Then: ValueError가 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        # When & Then
        with self.assertRaises(ValueError) as context:
            await embedder.embed_text("")

        self.assertIn("텍스트가 비어있거나 None입니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_text_none_text(self, mock_get_logger, mock_sentence_transformer):
        """Given: None 텍스트
        When: embed_text를 호출할 때
        Then: ValueError가 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        # When & Then
        with self.assertRaises(ValueError) as context:
            await embedder.embed_text(None)

        self.assertIn("텍스트가 비어있거나 None입니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_text_model_error(self, mock_get_logger, mock_sentence_transformer):
        """Given: 모델 인코딩 중 오류 발생
        When: embed_text를 호출할 때
        Then: HuggingFaceEmbeddingException이 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        original_error = Exception("Model encoding failed")
        self.mock_model.encode.side_effect = original_error

        # When & Then
        with self.assertRaises(HuggingFaceEmbeddingException) as context:
            await embedder.embed_text("test text")

        # 예외 내용 검증
        exception = context.exception
        self.assertEqual(exception.model_name, "test-model")
        self.assertEqual(exception.text, "test text")
        self.assertEqual(exception.error_code, "HUGGINGFACE_EMBEDDING_FAILED")
        self.assertEqual(exception.original_error, original_error)
        self.assertIn("텍스트 임베딩 중 오류가 발생했습니다", str(exception))

        self.mock_logger.error.assert_called()

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_texts_success_multiple(self, mock_get_logger, mock_sentence_transformer):
        """Given: 정상적으로 초기화된 임베더와 여러 텍스트
        When: embed_texts를 호출할 때
        Then: 각 텍스트에 대한 EmbeddingResult 리스트가 반환되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        texts = ["Hello", "World", "Test"]
        expected_embeddings = np.random.rand(3, 384).astype(np.float32)
        self.mock_model.encode.return_value = expected_embeddings

        # When
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.003]  # 시작 시간, 종료 시간
            results = await embedder.embed_texts(texts)

        # Then
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertIsInstance(result, EmbeddingResult)
            self.assertEqual(result.text, texts[i])
            self.assertEqual(result.embedding, expected_embeddings[i].tolist())
            self.assertEqual(result.model_name, "test-model")
            self.assertEqual(result.dimension, 384)
            self.assertEqual(result.processing_time_ms, 1.0)  # 평균 처리 시간
            self.assertEqual(result.metadata["batch_index"], i)
            self.assertEqual(result.metadata["batch_size"], 3)

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_texts_success_single(self, mock_get_logger, mock_sentence_transformer):
        """Given: 정상적으로 초기화된 임베더와 단일 텍스트 리스트
        When: embed_texts를 호출할 때
        Then: 1D 배열 처리가 정상적으로 되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        texts = ["Hello"]
        expected_embedding = np.random.rand(384).astype(np.float32)  # 1D 배열
        self.mock_model.encode.return_value = expected_embedding

        # When
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.001]  # 시작 시간, 종료 시간
            results = await embedder.embed_texts(texts)

        # Then
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.embedding, expected_embedding.tolist())
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.dimension, 384)

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_texts_empty_list(self, mock_get_logger, mock_sentence_transformer):
        """Given: 빈 텍스트 리스트
        When: embed_texts를 호출할 때
        Then: ValueError가 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        # When & Then
        with self.assertRaises(ValueError) as context:
            await embedder.embed_texts([])

        self.assertIn("텍스트 리스트가 비어있습니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_texts_invalid_text_in_list(
        self, mock_get_logger, mock_sentence_transformer
    ):
        """Given: 빈 문자열이나 None을 포함한 텍스트 리스트
        When: embed_texts를 호출할 때
        Then: ValueError가 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        # When & Then
        with self.assertRaises(ValueError) as context:
            await embedder.embed_texts(["Hello", "", "World"])

        self.assertIn("인덱스 [1]의 텍스트가 비어있거나 None입니다", str(context.exception))

        with self.assertRaises(ValueError) as context:
            await embedder.embed_texts(["Hello", None, "World"])

        self.assertIn("인덱스 [1]의 텍스트가 비어있거나 None입니다", str(context.exception))

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_embed_texts_model_error(self, mock_get_logger, mock_sentence_transformer):
        """Given: 모델 인코딩 중 오류 발생
        When: embed_texts를 호출할 때
        Then: HuggingFaceEmbeddingException이 발생해야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        original_error = Exception("Batch encoding failed")
        self.mock_model.encode.side_effect = original_error

        # When & Then
        with self.assertRaises(HuggingFaceEmbeddingException) as context:
            await embedder.embed_texts(["Hello", "World"])

        # 예외 내용 검증
        exception = context.exception
        self.assertEqual(exception.model_name, "test-model")
        self.assertEqual(exception.text, "Hello")  # 첫 번째 텍스트가 대표로 사용됨
        self.assertEqual(exception.error_code, "HUGGINGFACE_EMBEDDING_FAILED")
        self.assertEqual(exception.original_error, original_error)
        self.assertEqual(exception.context["batch_size"], 2)
        self.assertIn("텍스트 일괄 임베딩 중 오류가 발생했습니다", str(exception))

        self.mock_logger.error.assert_called()

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    def test_get_embedding_dimension(self, mock_get_logger, mock_sentence_transformer):
        """Given: 정상적으로 초기화된 임베더
        When: get_embedding_dimension을 호출할 때
        Then: 올바른 차원이 반환되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        # When
        dimension = embedder.get_embedding_dimension()

        # Then
        self.assertEqual(dimension, 384)

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_is_available_success(self, mock_get_logger, mock_sentence_transformer):
        """Given: 정상적으로 동작하는 임베더
        When: is_available을 호출할 때
        Then: True가 반환되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        test_embedding = np.random.rand(384).astype(np.float32)
        self.mock_model.encode.return_value = test_embedding

        # When
        is_available = await embedder.is_available()

        # Then
        self.assertTrue(is_available)
        self.mock_model.encode.assert_called_with(
            "test", convert_to_numpy=True, show_progress_bar=False
        )
        self.mock_logger.debug.assert_any_call("checking_service_availability")
        self.mock_logger.debug.assert_any_call("service_availability_check_passed")

    @patch("src.adapters.huggingface.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("src.adapters.huggingface.text_embedder.SentenceTransformer")
    @patch("src.adapters.huggingface.text_embedder.get_observable_logger")
    async def test_is_available_failure(self, mock_get_logger, mock_sentence_transformer):
        """Given: 오류가 발생하는 임베더
        When: is_available을 호출할 때
        Then: False가 반환되어야 함"""
        # Given
        mock_sentence_transformer.return_value = self.mock_model
        mock_get_logger.return_value = self.mock_logger
        embedder = HuggingFaceTextEmbedder("test-model")

        self.mock_model.encode.side_effect = Exception("Service unavailable")

        # When
        is_available = await embedder.is_available()

        # Then
        self.assertFalse(is_available)
        self.mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
