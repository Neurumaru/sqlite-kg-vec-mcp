"""
OllamaClient 엔티티 추출 기능 테스트.
"""

import json
import unittest
from unittest.mock import Mock, patch

from src.adapters.ollama.ollama_client import OllamaClient
from tests.unit.adapters.ollama.client.test_base import BaseOllamaClientTestCase


class TestOllamaClientEntityExtraction(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 엔티티 추출 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_success(self):
        """엔티티와 관계 추출 성공 테스트."""
        # Given: Successful extraction response
        extraction_result = {
            "nodes": [{"id": "1", "name": "Python", "type": "TECHNOLOGY"}],
            "relationships": [{"source": "1", "target": "2", "type": "USES"}],
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "response": f"```json\n{json.dumps(extraction_result)}\n```"
        }
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Extract entities and relationships
        with patch(
            "src.adapters.ollama.ollama_client.with_observability",
            lambda **kwargs: lambda func: func,
        ):  # Bypass decorator
            result = client.extract_entities_and_relationships("Python is a programming language")

        # Then: Should return extracted data
        self.assertEqual(result, extraction_result)

    def test_success_when_markdown_cleanup_needed(self):
        """마크다운 정리가 필요한 경우 엔티티 추출 테스트."""
        # Given: Response with markdown formatting
        extraction_result = {
            "nodes": [{"id": "1", "name": "AI", "type": "TECHNOLOGY"}],
            "relationships": [],
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "response": f"Here's the result:\n```json\n{json.dumps(extraction_result)}\n```\nDone!"
        }
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When: Extract entities and relationships
        with patch(
            "src.adapters.ollama.ollama_client.with_observability",
            lambda **kwargs: lambda func: func,
        ):  # Bypass decorator
            result = client.extract_entities_and_relationships("AI is artificial intelligence")

        # Then: Should return cleaned extraction data
        self.assertEqual(result, extraction_result)

    def test_value_error_when_json_error(self):
        """JSON 파싱 오류 시 ValueError 발생 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"response": "```json\ninvalid json\n```"}
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should raise ValueError
        with patch(
            "src.adapters.ollama.ollama_client.with_observability",
            lambda **kwargs: lambda func: func,
        ):  # Bypass decorator
            with self.assertRaises(ValueError):
                client.extract_entities_and_relationships("Some text")


class TestOllamaClientGenerateEmbeddings(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 임베딩 생성 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_success(self):
        """임베딩 생성 설명 테스트."""
        # Given: OllamaClient instance
        client = OllamaClient()

        # When: Get embeddings description
        test_entity = {"name": "test", "type": "CONCEPT"}
        description = client.generate_embeddings_description(test_entity)

        # Then: Should return proper description
        expected_description = (
            "Ollama 클라이언트는 직접적인 임베딩 생성을 지원하지 않습니다. "
            "별도의 임베딩 모델 어댑터를 사용하세요."
        )
        self.assertEqual(description, expected_description)


if __name__ == "__main__":
    unittest.main()
