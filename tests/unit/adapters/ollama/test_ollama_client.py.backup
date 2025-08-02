"""
Ollama Client 어댑터 단위 테스트.
"""

# pylint: disable=protected-access
import json
import unittest
from unittest.mock import Mock, patch

import requests

from src.adapters.ollama.exceptions import (
    OllamaConnectionException,
    OllamaResponseException,
    OllamaTimeoutException,
)
from src.adapters.ollama.ollama_client import LLMResponse, OllamaClient
from src.common.config.llm import OllamaConfig


class TestLLMResponse(unittest.TestCase):
    """LLMResponse 데이터클래스 테스트."""

    def test_llm_response_creation(self):
        """LLMResponse 생성 테스트."""
        # Given: Response data
        response = LLMResponse(
            text="Generated text",
            model="llama3.2",
            tokens_used=150,
            response_time=2.5,
            metadata={"temperature": 0.7},
        )

        # Then: All fields should be set correctly
        self.assertEqual(response.text, "Generated text")
        self.assertEqual(response.model, "llama3.2")
        self.assertEqual(response.tokens_used, 150)
        self.assertEqual(response.response_time, 2.5)
        self.assertEqual(response.metadata, {"temperature": 0.7})

    def test_llm_response_default_metadata(self):
        """LLMResponse 기본 메타데이터 테스트."""
        # Given: Response data without metadata
        response = LLMResponse(
            text="Generated text", model="llama3.2", tokens_used=150, response_time=2.5
        )

        # Then: Metadata should be None
        self.assertIsNone(response.metadata)


class TestOllamaClient(unittest.TestCase):
    """OllamaClient 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # Mock requests.Session to avoid actual HTTP calls
        self.mock_session_patcher = patch("src.adapters.ollama.ollama_client.requests.Session")
        mock_session_cls = self.mock_session_patcher.start()
        self.mock_session = Mock()
        mock_session_cls.return_value = self.mock_session

        # Mock successful connection test
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_session.get.return_value = mock_response

        # Mock logger
        self.mock_logger_patcher = patch("src.adapters.ollama.ollama_client.get_observable_logger")
        mock_logger_fn = self.mock_logger_patcher.start()
        self.mock_logger = Mock()
        mock_logger_fn.return_value = self.mock_logger

    def tearDown(self):
        """테스트 정리."""
        self.mock_session_patcher.stop()
        self.mock_logger_patcher.stop()

    def test_initialization_with_config(self):
        """설정 객체로 초기화 테스트."""
        # Given: Ollama configuration
        config = OllamaConfig(
            host="test-host",
            port=8080,
            model="test-model",
            temperature=0.5,
            max_tokens=1000,
            timeout=45.0,
        )

        # When: Create client with config
        client = OllamaClient(config=config)

        # Then: Client should be configured correctly
        self.assertEqual(client.base_url, "http://test-host:8080")
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.timeout, 45)
        self.assertEqual(client.temperature, 0.5)
        self.assertEqual(client.max_tokens, 1000)

    def test_initialization_with_individual_params(self):
        """개별 파라미터로 초기화 테스트 (하위 호환성)."""
        # Given: Individual parameters
        client = OllamaClient(base_url="http://custom:9000", model="custom-model", timeout=30)

        # Then: Client should use individual parameters
        self.assertEqual(client.base_url, "http://custom:9000")
        self.assertEqual(client.model, "custom-model")
        self.assertEqual(client.timeout, 30)

    def test_initialization_default_config(self):
        """기본 설정으로 초기화 테스트."""
        # When: Create client without config
        client = OllamaClient()

        # Then: Should use default config values
        self.assertEqual(client.base_url, "http://localhost:11434")
        # Model value comes from environment variables, so we just check it's set
        self.assertIsNotNone(client.model)
        self.assertEqual(client.timeout, 30)

    def test_test_connection_success(self):
        """연결 테스트 성공 케이스."""
        # Given: Successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_session.get.return_value = mock_response

        # When: Test connection
        client = OllamaClient()
        result = client._test_connection()

        # Then: Should return True
        self.assertTrue(result)
        self.mock_session.get.assert_called_with("http://localhost:11434/api/tags", timeout=5)

    def test_test_connection_connection_error(self):
        """연결 테스트 연결 오류 케이스."""
        # Given: Connection error
        self.mock_session.get.side_effect = requests.ConnectionError("Connection failed")

        # When: Test connection
        client = OllamaClient()
        result = client._test_connection()

        # Then: Should return False and log warning
        self.assertFalse(result)
        self.mock_logger.warning.assert_called()

    def test_test_connection_timeout(self):
        """연결 테스트 타임아웃 케이스."""
        # Given: Timeout error
        self.mock_session.get.side_effect = requests.Timeout("Request timed out")

        # When: Test connection
        client = OllamaClient()
        result = client._test_connection()

        # Then: Should return False and log warning
        self.assertFalse(result)
        self.mock_logger.warning.assert_called()

    def test_test_connection_http_error(self):
        """연결 테스트 HTTP 오류 케이스."""
        # Given: HTTP error
        http_error = requests.HTTPError("HTTP 500 Error")
        mock_response = Mock()
        mock_response.status_code = 500
        http_error.response = mock_response
        self.mock_session.get.side_effect = http_error

        # When: Test connection
        client = OllamaClient()
        result = client._test_connection()

        # Then: Should return False and log warning
        self.assertFalse(result)
        self.mock_logger.warning.assert_called()

    def test_generate_success_non_streaming(self):
        """텍스트 생성 성공 테스트 (비스트리밍)."""
        # Given: Successful generation response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"response": "Generated text response", "eval_count": 25}
        self.mock_session.post.return_value = mock_response

        client = OllamaClient(model="llama3.2")

        # When: Generate text
        mock_time_counter = [1000.0]  # Use a list to make it mutable within lambda
        with (
            patch(
                "time.time",
                side_effect=lambda: mock_time_counter.__setitem__(0, mock_time_counter[0] + 1.0)
                or mock_time_counter[0],
            ),  # Mock timing
            patch(
                "src.adapters.ollama.ollama_client.with_observability",
                lambda **kwargs: lambda func: func,
            ),  # Bypass decorator
        ):
            result = client.generate(
                prompt="Test prompt",
                system_prompt="System instruction",
                temperature=0.8,
                max_tokens=100,
            )

        # Then: Should return LLMResponse with correct data
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.text, "Generated text response")
        self.assertEqual(result.model, "llama3.2")
        self.assertEqual(result.tokens_used, 25)
        self.assertEqual(result.response_time, 1.0)
        self.assertEqual(result.metadata, {"temperature": 0.8})

        # Verify API call
        expected_data = {
            "model": client.model,
            "prompt": "Test prompt",
            "stream": False,
            "system": "System instruction",
            "options": {"temperature": 0.8, "num_predict": 100},
        }
        self.mock_session.post.assert_called_with(
            "http://localhost:11434/api/generate", json=expected_data, timeout=30
        )

    def test_generate_success_streaming(self):
        """텍스트 생성 성공 테스트 (스트리밍)."""
        # Given: Streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""

        # Mock streaming chunks
        streaming_data = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": false}',
            b'{"response": "!", "done": true, "eval_count": 15}',
        ]
        mock_response.iter_lines.return_value = streaming_data
        self.mock_session.post.return_value = mock_response

        client = OllamaClient(model="llama3.2")

        # When: Generate text with streaming
        mock_time_counter = [1000.0]  # Use a list to make it mutable within lambda
        with (
            patch(
                "time.time",
                side_effect=lambda: mock_time_counter.__setitem__(0, mock_time_counter[0] + 0.5)
                or mock_time_counter[0],
            ),  # Mock timing
            patch(
                "src.adapters.ollama.ollama_client.with_observability",
                lambda **kwargs: lambda func: func,
            ),  # Bypass decorator
        ):
            result = client.generate(prompt="Test prompt", stream=True)

        # Then: Should return combined streaming response
        self.assertEqual(result.text, "Hello world!")
        self.assertEqual(result.tokens_used, 15)
        self.assertEqual(result.response_time, 0.5)

    def test_generate_connection_error(self):
        """텍스트 생성 연결 오류 테스트."""
        # Given: Connection error
        self.mock_session.post.side_effect = requests.ConnectionError("Connection failed")

        client = OllamaClient()

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.generate("Test prompt")

    def test_generate_timeout_error(self):
        """텍스트 생성 타임아웃 오류 테스트."""
        # Given: Timeout error
        self.mock_session.post.side_effect = requests.Timeout("Request timed out")

        client = OllamaClient()

        # When & Then: Should raise OllamaTimeoutException
        with self.assertRaises(OllamaTimeoutException):
            client.generate("Test prompt")

    def test_generate_http_error(self):
        """텍스트 생성 HTTP 오류 테스트."""
        # Given: HTTP error
        http_error = requests.HTTPError("HTTP 500 Error")
        mock_response = Mock()
        mock_response.status_code = 500
        http_error.response = mock_response
        self.mock_session.post.side_effect = http_error

        client = OllamaClient()

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.generate("Test prompt")

    def test_generate_json_decode_error(self):
        """텍스트 생성 JSON 파싱 오류 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Invalid JSON"
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should raise OllamaResponseException
        with self.assertRaises(OllamaResponseException):
            client.generate("Test prompt")

    def test_generate_data_processing_error(self):
        """텍스트 생성 데이터 처리 오류 테스트."""
        # Given: Response with missing expected fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {}  # Empty response
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should handle gracefully and not raise exception
        result = client.generate("Test prompt")

        # Should return response with empty text and zero tokens
        self.assertEqual(result.text, "")
        self.assertEqual(result.tokens_used, 0)

    def test_extract_entities_and_relationships_success(self):
        """엔티티 및 관계 추출 성공 테스트."""
        # Given: Successful LLM response with valid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "response": '{"entities": [{"id": "1", "name": "John", "type": "Person"}], "relationships": []}',
            "eval_count": 50,
        }
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Extract entities and relationships
        result = client.extract_entities_and_relationships("John is a person.")

        # Then: Should return parsed structure
        expected = {
            "entities": [{"id": "1", "name": "John", "type": "Person"}],
            "relationships": [],
        }
        self.assertEqual(result, expected)

    def test_extract_entities_and_relationships_markdown_cleanup(self):
        """마크다운 코드 블록 정리 테스트."""
        # Given: Response with markdown code blocks
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "response": '```json\n{"entities": [], "relationships": []}\n```',
            "eval_count": 30,
        }
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Extract entities and relationships
        result = client.extract_entities_and_relationships("Test text")

        # Then: Should clean markdown and parse JSON
        expected = {"entities": [], "relationships": []}
        self.assertEqual(result, expected)

    def test_extract_entities_and_relationships_json_error(self):
        """JSON 파싱 오류 시 빈 구조 반환 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"response": "Invalid JSON response", "eval_count": 20}
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Extract entities and relationships
        result = client.extract_entities_and_relationships("Test text")

        # Then: Should return empty structure
        expected = {"entities": [], "relationships": []}
        self.assertEqual(result, expected)
        self.mock_logger.error.assert_called()

    def test_generate_embeddings_description(self):
        """임베딩 설명 생성 테스트."""
        # Given: Successful description generation
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "response": "John is a software engineer with expertise in Python.",
            "eval_count": 12,
        }
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Generate embeddings description
        entity = {"name": "John", "type": "Person", "properties": {"occupation": "engineer"}}
        result = client.generate_embeddings_description(entity)

        # Then: Should return description
        self.assertEqual(result, "John is a software engineer with expertise in Python.")

    def test_list_available_models_success(self):
        """사용 가능한 모델 목록 조회 성공 테스트."""
        # Given: Successful models response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "codellama"}, {"name": "mistral"}]
        }
        self.mock_session.get.return_value = mock_response

        client = OllamaClient()

        # When: List available models
        models = client.list_available_models()

        # Then: Should return model names
        expected = ["llama3.2", "codellama", "mistral"]
        self.assertEqual(models, expected)

    def test_list_available_models_error(self):
        """모델 목록 조회 오류 테스트."""
        # Given: Request error
        self.mock_session.get.side_effect = requests.RequestException("API Error")

        client = OllamaClient()

        # When: List available models
        models = client.list_available_models()

        # Then: Should return empty list and log error
        self.assertEqual(models, [])
        self.mock_logger.error.assert_called()

    def test_pull_model_success(self):
        """모델 다운로드 성공 테스트."""
        # Given: Successful pull response
        mock_response = Mock()
        mock_response.status_code = 200

        # Mock streaming response with success status
        streaming_data = [b'{"status": "downloading"}', b'{"status": "success"}']
        mock_response.iter_lines.return_value = streaming_data
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Pull model
        result = client.pull_model("new-model")

        # Then: Should return True
        self.assertTrue(result)
        self.mock_session.post.assert_called_with(
            "http://localhost:11434/api/pull", json={"name": "new-model"}, timeout=300
        )

    def test_pull_model_failure(self):
        """모델 다운로드 실패 테스트."""
        # Given: Failed pull response
        mock_response = Mock()
        mock_response.status_code = 200

        # Mock streaming response without success status
        streaming_data = [
            b'{"status": "downloading"}',
            b'{"status": "error", "error": "Model not found"}',
        ]
        mock_response.iter_lines.return_value = streaming_data
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Pull model
        result = client.pull_model("non-existent-model")

        # Then: Should return False
        self.assertFalse(result)

    def test_pull_model_request_error(self):
        """모델 다운로드 요청 오류 테스트."""
        # Given: Request error
        self.mock_session.post.side_effect = requests.RequestException("Network error")

        client = OllamaClient()

        # When: Pull model
        result = client.pull_model("test-model")

        # Then: Should return False and log error
        self.assertFalse(result)
        self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
