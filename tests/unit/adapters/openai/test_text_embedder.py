"""
OpenAI TextEmbedder 어댑터 단위 테스트.
"""

# pylint: disable=protected-access

import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.adapters.openai.text_embedder import OpenAITextEmbedder
from src.common.config.llm import OpenAIConfig
from src.dto.embedding import EmbeddingResult


class TestOpenAITextEmbedder(unittest.IsolatedAsyncioTestCase):
    """OpenAI TextEmbedder 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # Mock OpenAI availability and library
        self.openai_patcher = patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True)
        self.openai_patcher.start()

        # Mock openai module and create custom exception classes
        class MockAuthenticationError(Exception):
            """Mock OpenAI 인증 오류."""

        class MockRateLimitError(Exception):
            """Mock OpenAI 요청 제한 오류."""

        class MockAPIConnectionError(Exception):
            """Mock OpenAI API 연결 오류."""

        class MockAPIError(Exception):
            """Mock OpenAI API 일반 오류."""

        self.openai_mock = Mock()
        self.openai_mock.AuthenticationError = MockAuthenticationError
        self.openai_mock.RateLimitError = MockRateLimitError
        self.openai_mock.APIConnectionError = MockAPIConnectionError
        self.openai_mock.APIError = MockAPIError

        with patch("src.adapters.openai.text_embedder.openai", self.openai_mock):
            with patch("src.adapters.openai.text_embedder.AsyncOpenAI") as mock_async_openai:
                self.mock_client = Mock()
                mock_async_openai.return_value = self.mock_client

                self.config = OpenAIConfig(api_key="sk-test-api-key")
                self.embedder = OpenAITextEmbedder(config=self.config)

    def tearDown(self):
        """테스트 후 정리."""
        self.openai_patcher.stop()

    # === 초기화 테스트 ===

    def test_initialization_with_config(self):
        """설정 객체를 통한 초기화 테스트."""
        # Given
        config = OpenAIConfig(
            api_key="sk-test-config-key",
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
            timeout=60.0,
        )

        # When
        with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True):
            with patch("src.adapters.openai.text_embedder.openai"):
                with patch("src.adapters.openai.text_embedder.AsyncOpenAI"):
                    embedder = OpenAITextEmbedder(config=config)

        # Then
        self.assertEqual(embedder.api_key, "sk-test-config-key")
        self.assertEqual(embedder.model, "text-embedding-3-large")
        self.assertEqual(embedder.custom_dimension, 3072)
        self.assertEqual(embedder.timeout, 60.0)
        self.assertEqual(embedder._dimension, 3072)

    def test_initialization_with_individual_parameters(self):
        """개별 파라미터를 통한 초기화 테스트 (하위 호환성)."""
        # Given & When
        with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True):
            with patch("src.adapters.openai.text_embedder.openai"):
                with patch("src.adapters.openai.text_embedder.AsyncOpenAI"):
                    embedder = OpenAITextEmbedder(
                        api_key="sk-test-individual-key",
                        model="text-embedding-ada-002",
                        dimension=1536,
                    )

        # Then
        self.assertEqual(embedder.api_key, "sk-test-individual-key")
        self.assertEqual(embedder.model, "text-embedding-ada-002")
        self.assertEqual(embedder.custom_dimension, 1536)
        self.assertEqual(embedder._dimension, 1536)

    def test_initialization_without_openai_library_raises_error(self):
        """OpenAI 라이브러리 없이 초기화 시 에러 발생 테스트."""
        # Given & When & Then
        with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", False):
            with self.assertRaises(ImportError) as context:
                OpenAITextEmbedder(api_key="sk-test-key")

            self.assertIn("openai가 설치되지 않았습니다", str(context.exception))

    def test_initialization_without_api_key_raises_error(self):
        """API 키 없이 초기화 시 에러 발생 테스트."""
        # Given
        config = OpenAIConfig(api_key=None)

        # When & Then
        with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True):
            with patch("src.adapters.openai.text_embedder.openai"):
                with patch.dict("os.environ", {}, clear=True):
                    with self.assertRaises(ValueError) as context:
                        OpenAITextEmbedder(config=config)

                    self.assertIn("OpenAI API 키가 제공되지 않았고", str(context.exception))

    def test_initialization_default_model_dimensions(self):
        """기본 모델 차원 설정 테스트."""
        # Given
        test_cases = [
            ("text-embedding-ada-002", 1536),
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("unknown-model", 1536),  # 기본값
        ]

        # When & Then
        for model, expected_dimension in test_cases:
            with self.subTest(model=model):
                config = OpenAIConfig(api_key="sk-test-key", embedding_model=model)

                with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True):
                    with patch("src.adapters.openai.text_embedder.openai"):
                        with patch("src.adapters.openai.text_embedder.AsyncOpenAI"):
                            embedder = OpenAITextEmbedder(config=config)

                self.assertEqual(embedder._dimension, expected_dimension)

    def test_get_embedding_dimension(self):
        """임베딩 차원 반환 테스트."""
        # Given & When
        dimension = self.embedder.get_embedding_dimension()

        # Then
        self.assertEqual(dimension, 1536)  # text-embedding-3-small 기본값

    # === embed_text 메서드 테스트 ===

    async def test_embed_text_success(self):
        """단일 텍스트 임베딩 성공 테스트."""
        # Given
        test_text = "테스트 텍스트"
        mock_embedding = [0.1, 0.2, 0.3, 0.4]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=mock_embedding)]
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 10}
        mock_response.model = "text-embedding-3-small"

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When
        result = await self.embedder.embed_text(test_text)

        # Then
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(result.text, test_text)
        # 부동소수점 정밀도 차이를 고려한 비교
        for actual, expected in zip(result.embedding, mock_embedding, strict=False):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(result.model_name, "text-embedding-3-small")
        self.assertEqual(result.dimension, 4)
        self.assertIsNotNone(result.processing_time_ms)
        self.assertGreater(result.processing_time_ms, 0)
        self.assertIn("usage", result.metadata)
        self.assertIn("model", result.metadata)

    async def test_embed_text_with_custom_dimension(self):
        """커스텀 차원으로 텍스트 임베딩 테스트."""
        # Given
        config = OpenAIConfig(api_key="sk-test-key", embedding_dimension=512)

        with patch("src.adapters.openai.text_embedder.OPENAI_AVAILABLE", True):
            with patch("src.adapters.openai.text_embedder.openai"):
                with patch("src.adapters.openai.text_embedder.AsyncOpenAI") as mock_async_openai:
                    mock_client = Mock()
                    mock_async_openai.return_value = mock_client

                    embedder = OpenAITextEmbedder(config=config)

                    mock_response = Mock()
                    mock_response.data = [Mock(embedding=[0.1] * 512)]
                    mock_response.usage = Mock()
                    mock_response.usage.model_dump.return_value = {}

                    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

                    # When
                    result = await embedder.embed_text("테스트")

                    # Then
                    self.assertEqual(result.dimension, 512)
                    # API 호출 시 dimensions 파라미터가 전달되었는지 확인
                    mock_client.embeddings.create.assert_called_once()
                    call_args = mock_client.embeddings.create.call_args[1]
                    self.assertEqual(call_args["dimensions"], 512)

    async def test_embed_text_empty_string_raises_error(self):
        """빈 문자열로 임베딩 시 에러 발생 테스트."""
        # Given
        empty_texts = ["", "   ", "\t\n"]

        # When & Then
        for empty_text in empty_texts:
            with self.subTest(text=repr(empty_text)):
                with self.assertRaises(ValueError) as context:
                    await self.embedder.embed_text(empty_text)

                self.assertIn("텍스트가 비어있습니다", str(context.exception))

    async def test_embed_text_api_authentication_error(self):
        """API 인증 실패 시 에러 처리 테스트."""
        # Given - 실제 API 예외를 시뮬레이션
        self.mock_client.embeddings.create = AsyncMock(
            side_effect=Exception("Authentication failed")
        )

        # When & Then - 예상치 못한 오류로 처리됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))

    async def test_embed_text_api_rate_limit_error(self):
        """API 요청 한도 초과 시 에러 처리 테스트."""
        # Given - 일반적인 API 에러 시뮬레이션
        self.mock_client.embeddings.create = AsyncMock(side_effect=Exception("Rate limit exceeded"))

        # When & Then - 예상치 못한 오류로 처리됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))

    async def test_embed_text_api_connection_error(self):
        """API 연결 실패 시 에러 처리 테스트."""
        # Given - 연결 에러 시뮬레이션
        self.mock_client.embeddings.create = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )

        # When & Then - 예상치 못한 오류로 처리됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))

    async def test_embed_text_general_api_error(self):
        """일반 API 에러 처리 테스트."""
        # Given - 일반적인 예외 발생
        self.mock_client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("General API error")
        )

        # When & Then - 예상치 못한 오류로 처리됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))

    async def test_embed_text_unexpected_error(self):
        """예상치 못한 에러 처리 테스트."""
        # Given
        self.mock_client.embeddings.create = AsyncMock(side_effect=KeyError("Unexpected error"))

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))

    async def test_embed_text_empty_response_data(self):
        """API 응답에 데이터가 없을 때 에러 처리 테스트."""
        # Given
        mock_response = Mock()
        mock_response.data = []
        mock_response.usage = None

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When & Then - ValueError가 RuntimeError로 래핑됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_text("테스트")

        self.assertIn("임베딩 생성 중 예상치 못한 오류", str(context.exception))
        self.assertIn("OpenAI API에서 임베딩 데이터를 받지 못했습니다", str(context.exception))

    # === embed_texts 메서드 테스트 ===

    async def test_embed_texts_success(self):
        """여러 텍스트 일괄 임베딩 성공 테스트."""
        # Given
        test_texts = ["텍스트1", "텍스트2", "텍스트3"]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 30}
        mock_response.model = "text-embedding-3-small"

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When
        results = await self.embedder.embed_texts(test_texts)

        # Then
        self.assertEqual(len(results), 3)

        for i, (result, expected_text, expected_embedding) in enumerate(
            zip(results, test_texts, mock_embeddings, strict=False)
        ):
            with self.subTest(index=i):
                self.assertIsInstance(result, EmbeddingResult)
                self.assertEqual(result.text, expected_text)
                # 부동소수점 정밀도 차이를 고려한 비교
                for actual, expected in zip(result.embedding, expected_embedding, strict=False):
                    self.assertAlmostEqual(actual, expected, places=5)
                self.assertEqual(result.dimension, 2)
                self.assertEqual(result.model_name, "text-embedding-3-small")
                self.assertIn("batch_index", result.metadata)
                self.assertEqual(result.metadata["batch_index"], i)
                self.assertEqual(result.metadata["batch_size"], 3)

    async def test_embed_texts_empty_list(self):
        """빈 리스트로 일괄 임베딩 시 빈 결과 반환 테스트."""
        # Given & When
        results = await self.embedder.embed_texts([])

        # Then
        self.assertEqual(results, [])

    async def test_embed_texts_with_empty_string_raises_error(self):
        """빈 문자열이 포함된 리스트로 임베딩 시 에러 발생 테스트."""
        # Given
        test_texts = ["정상 텍스트", "", "또 다른 정상 텍스트"]

        # When & Then
        with self.assertRaises(ValueError) as context:
            await self.embedder.embed_texts(test_texts)

        self.assertIn("인덱스 1의 텍스트가 비어있습니다", str(context.exception))

    async def test_embed_texts_mismatched_response_count(self):
        """API 응답 개수가 요청과 일치하지 않을 때 에러 처리 테스트."""
        # Given
        test_texts = ["텍스트1", "텍스트2", "텍스트3"]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]  # 요청 3개, 응답 1개
        mock_response.usage = None

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When & Then - ValueError가 RuntimeError로 래핑됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_texts(test_texts)

        self.assertIn("일괄 임베딩 생성 중 예상치 못한 오류", str(context.exception))
        self.assertIn("예상한 수만큼의 임베딩 데이터를 받지 못했습니다", str(context.exception))

    async def test_embed_texts_general_error(self):
        """일괄 임베딩 시 예외 처리 테스트."""
        # Given - 일반적인 예외 발생
        self.mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))

        # When & Then - 예상치 못한 오류로 처리됨
        with self.assertRaises(RuntimeError) as context:
            await self.embedder.embed_texts(["텍스트1", "텍스트2"])

        self.assertIn("일괄 임베딩 생성 중 예상치 못한 오류", str(context.exception))

    # === is_available 메서드 테스트 ===

    async def test_is_available_success(self):
        """서비스 가용성 확인 성공 테스트."""
        # Given
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {}

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertTrue(result)

    async def test_is_available_value_error(self):
        """ValueError 발생 시 서비스 사용 불가 처리 테스트."""
        # Given
        self.mock_client.embeddings.create = AsyncMock(side_effect=ValueError("API error"))

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertFalse(result)

    async def test_is_available_connection_error(self):
        """ConnectionError 발생 시 서비스 사용 불가 처리 테스트."""
        # Given
        self.mock_client.embeddings.create = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertFalse(result)

    async def test_is_available_runtime_error(self):
        """RuntimeError 발생 시 서비스 사용 불가 처리 테스트."""
        # Given
        self.mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("Runtime error"))

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertFalse(result)

    async def test_is_available_unexpected_error(self):
        """예상치 못한 에러 발생 시 서비스 사용 불가 처리 테스트."""
        # Given
        self.mock_client.embeddings.create = AsyncMock(side_effect=KeyError("Unexpected error"))

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertFalse(result)

    async def test_is_available_empty_embedding(self):
        """빈 임베딩 응답 시 서비스 사용 불가 처리 테스트."""
        # Given
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[])]
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {}

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When
        result = await self.embedder.is_available()

        # Then
        self.assertFalse(result)

    # === 통합 테스트 ===

    async def test_full_workflow_integration(self):
        """전체 워크플로우 통합 테스트."""
        # Given
        test_text = "통합 테스트 텍스트"
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=mock_embedding)]
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 5}
        mock_response.model = "text-embedding-3-small"

        self.mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        # When
        # 1. 서비스 가용성 확인
        is_available = await self.embedder.is_available()

        # 2. 차원 정보 확인
        dimension = self.embedder.get_embedding_dimension()

        # 3. 단일 텍스트 임베딩
        single_result = await self.embedder.embed_text(test_text)

        # 4. 여러 텍스트 임베딩
        mock_response.data = [Mock(embedding=mock_embedding), Mock(embedding=mock_embedding)]
        multiple_results = await self.embedder.embed_texts([test_text, test_text])

        # Then
        self.assertTrue(is_available)
        self.assertEqual(dimension, 1536)

        self.assertIsInstance(single_result, EmbeddingResult)
        self.assertEqual(single_result.text, test_text)
        # 부동소수점 정밀도 차이를 고려한 비교
        for actual, expected in zip(single_result.embedding, mock_embedding, strict=False):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(single_result.dimension, 5)

        self.assertEqual(len(multiple_results), 2)
        for result in multiple_results:
            self.assertIsInstance(result, EmbeddingResult)
            self.assertEqual(result.text, test_text)
            # 부동소수점 정밀도 차이를 고려한 비교
            for actual, expected in zip(result.embedding, mock_embedding, strict=False):
                self.assertAlmostEqual(actual, expected, places=5)


if __name__ == "__main__":
    unittest.main()
