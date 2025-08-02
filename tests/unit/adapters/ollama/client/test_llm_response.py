"""
LLMResponse 데이터클래스 단위 테스트.
"""

import unittest

from src.adapters.ollama.ollama_client import LLMResponse


class TestLLMResponse(unittest.TestCase):
    """LLMResponse 데이터클래스 테스트."""

    def test_success(self):
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

    def test_success_when_default_metadata(self):
        """LLMResponse 기본 메타데이터 테스트."""
        # Given: Response data without metadata
        response = LLMResponse(
            text="Generated text", model="llama3.2", tokens_used=150, response_time=2.5
        )

        # Then: Metadata should be None
        self.assertIsNone(response.metadata)


if __name__ == "__main__":
    unittest.main()
