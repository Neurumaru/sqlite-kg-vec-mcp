"""
OllamaClient 모델 관리 기능 테스트.
"""

import unittest
from unittest.mock import Mock

import requests

from src.adapters.ollama.exceptions import OllamaConnectionException
from src.adapters.ollama.ollama_client import OllamaClient
from tests.unit.adapters.ollama.client.test_base import BaseOllamaClientTestCase


class TestOllamaClientModelManagement(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 모델 관리 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_list_available_models_success(self):
        """사용 가능한 모델 목록 조회 성공 테스트."""
        # Given: Successful model list response
        models_data = {
            "models": [
                {"name": "llama3.2:latest", "size": 2000000000},
                {"name": "codellama:7b", "size": 3800000000},
            ]
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = models_data
        self.mock_session.get.return_value = mock_response

        client = OllamaClient()

        # When: List available models
        result = client.list_available_models()

        # Then: Should return models list
        expected_models = ["llama3.2:latest", "codellama:7b"]
        self.assertEqual(result, expected_models)
        self.mock_session.get.assert_called_with("http://localhost:11434/api/tags", timeout=5.0)

    def test_list_available_models_error(self):
        """모델 목록 조회 오류 테스트."""
        # Given: Connection error
        self.mock_session.get.side_effect = requests.ConnectionError("Connection failed")

        client = OllamaClient()

        # When: Call list_available_models during error
        result = client.list_available_models()
        
        # Then: Should return empty list on error
        self.assertEqual(result, [])

    def test_pull_model_success(self):
        """모델 다운로드 성공 테스트."""
        # Given: Successful pull response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        # Mock iter_lines to return success status
        mock_response.iter_lines.return_value = [b'{"status": "success"}']
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Pull model
        result = client.pull_model("llama3.2")

        # Then: Should return success status
        self.assertTrue(result)
        expected_data = {"name": "llama3.2"}
        self.mock_session.post.assert_called_with(
            "http://localhost:11434/api/pull", json=expected_data, timeout=300.0
        )

    def test_pull_model_failure(self):
        """모델 다운로드 실패 테스트."""
        # Given: Failed pull response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Model not found"}
        # Mock iter_lines to return failure status
        mock_response.iter_lines.return_value = [b'{"status": "error", "error": "Model not found"}']
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Pull non-existent model
        result = client.pull_model("non-existent-model")

        # Then: Should return False
        self.assertFalse(result)

    def test_pull_model_request_error(self):
        """모델 다운로드 요청 오류 테스트."""
        # Given: Request error
        self.mock_session.post.side_effect = requests.ConnectionError("Connection failed")

        client = OllamaClient()

        # When: Call pull_model during error
        result = client.pull_model("llama3.2")
        
        # Then: Should return False on error
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
