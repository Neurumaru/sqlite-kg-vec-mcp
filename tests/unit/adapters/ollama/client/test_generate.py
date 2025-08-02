"""
OllamaClient 텍스트 생성 기능 테스트.
"""

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
from tests.unit.adapters.ollama.client.test_base import BaseOllamaClientTestCase


class TestOllamaClientGenerate(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 텍스트 생성 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_success_when_non_streaming(self):
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

    def test_success_when_streaming(self):
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

    def test_ollama_connection_exception_when_connection_error(self):
        """텍스트 생성 연결 오류 테스트."""
        # Given: Connection error
        self.mock_session.post.side_effect = requests.ConnectionError("Connection failed")

        client = OllamaClient()

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.generate("Test prompt")

    def test_ollama_timeout_exception_when_timeout_error(self):
        """텍스트 생성 타임아웃 오류 테스트."""
        # Given: Timeout error
        self.mock_session.post.side_effect = requests.Timeout("Timeout occurred")

        client = OllamaClient()

        # When & Then: Should raise OllamaTimeoutException
        with self.assertRaises(OllamaTimeoutException):
            client.generate("Test prompt")

    def test_ollama_connection_exception_when_http_error(self):
        """텍스트 생성 HTTP 오류 테스트."""
        # Given: HTTP error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.generate("Test prompt")

    def test_ollama_response_exception_when_json_decode_error(self):
        """텍스트 생성 JSON 디코딩 오류 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "invalid json"
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "invalid json", 0)
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should raise OllamaResponseException
        with self.assertRaises(OllamaResponseException):
            client.generate("Test prompt")

    def test_ollama_response_exception_when_data_processing_error(self):
        """텍스트 생성 데이터 처리 오류 테스트."""
        # Given: Response missing required fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {}  # Missing 'response' field
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should raise OllamaResponseException
        with self.assertRaises(OllamaResponseException):
            client.generate("Test prompt")


if __name__ == "__main__":
    unittest.main()
