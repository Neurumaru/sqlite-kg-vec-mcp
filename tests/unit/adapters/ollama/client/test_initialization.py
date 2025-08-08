"""
OllamaClient 초기화 기능 테스트.
"""

import unittest

from src.adapters.ollama.ollama_client import OllamaClient
from src.common.config.llm import OllamaConfig
from tests.unit.adapters.ollama.client.test_base import BaseOllamaClientTestCase


class TestOllamaClientInitialization(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 초기화 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_success_when_config_provided(self):
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
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.temperature, 0.5)
        self.assertEqual(client.max_tokens, 1000)

    def test_success_when_individual_params_provided(self):
        """개별 파라미터로 초기화 테스트."""
        # When: Create client with individual parameters
        client = OllamaClient(base_url="http://localhost:11434", model="llama3")

        # Then: Client should be configured correctly
        self.assertEqual(client.base_url, "http://localhost:11434")
        self.assertEqual(client.model, "llama3")

    def test_success_when_default_config(self):
        """기본 설정으로 초기화 테스트."""
        # When: Create client with defaults
        client = OllamaClient()

        # Then: Should use default values
        self.assertEqual(client.base_url, "http://localhost:11434")
        self.assertEqual(client.model, "gemma3n")


if __name__ == "__main__":
    unittest.main()
