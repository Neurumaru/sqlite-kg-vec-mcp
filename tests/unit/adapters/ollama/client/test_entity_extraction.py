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
            "entities": [],
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
            "entities": [],
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

    def test_default_structure_when_json_error(self):
        """JSON 파싱 오류 시 기본 구조 반환 테스트."""
        # Given: Invalid JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {"response": "```json\ninvalid json\n```"}
        self.mock_session.post.return_value = mock_response

        client = OllamaClient()

        # When & Then: Should return default structure for invalid JSON
        with patch(
            "src.adapters.ollama.ollama_client.with_observability",
            lambda **kwargs: lambda func: func,
        ):  # Bypass decorator
            result = client.extract_entities_and_relationships("Some text")
            
        # Should return default structure when JSON parsing fails
        expected_result = {"entities": [], "relationships": []}
        self.assertEqual(result, expected_result)


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
        # Given: Mock response for generate method
        from src.adapters.ollama.ollama_client import LLMResponse
        
        mock_response = LLMResponse(
            text="test는 컨셉 유형의 엔티티입니다.",
            model="gemma3n",
            tokens_used=10,
            response_time=0.1
        )
        
        # Mock the generate method
        with patch.object(OllamaClient, 'generate', return_value=mock_response):
            client = OllamaClient()

            # When: Get embeddings description
            test_entity = {"name": "test", "type": "CONCEPT"}
            description = client.generate_embeddings_description(test_entity)

            # Then: Should return generated description
            expected_description = "test는 컨셉 유형의 엔티티입니다."
            self.assertEqual(description, expected_description)


if __name__ == "__main__":
    unittest.main()
