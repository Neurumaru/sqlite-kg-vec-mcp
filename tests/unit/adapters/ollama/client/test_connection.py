"""
OllamaClient 연결 테스트 기능.
"""

import unittest
from unittest.mock import Mock

import requests

from src.adapters.ollama.exceptions import (
    OllamaConnectionException,
    OllamaTimeoutException,
)
from src.adapters.ollama.ollama_client import OllamaClient
from tests.unit.adapters.ollama.client.test_base import BaseOllamaClientTestCase


class TestOllamaClientConnection(unittest.TestCase, BaseOllamaClientTestCase):
    """OllamaClient 연결 테스트."""

    def setUp(self):
        """테스트 픽스처 설정."""
        BaseOllamaClientTestCase.setUp(self)

    def tearDown(self):
        """테스트 정리."""
        BaseOllamaClientTestCase.tearDown(self)

    def test_success(self):
        """연결 성공 테스트."""
        # Given: Successful response
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_session.get.return_value = mock_response

        client = OllamaClient()

        # Reset mock to test only test_connection method call
        self.mock_session.get.reset_mock()

        # When: Test connection
        result = client.test_connection()

        # Then: Should return True
        self.assertTrue(result)
        self.mock_session.get.assert_called_once()

    def test_ollama_connection_exception_when_connection_error(self):
        """연결 오류 시 예외 발생 테스트."""
        # Given: Successful response for constructor, then connection error
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_session.get.return_value = mock_response

        client = OllamaClient()

        # Now set up connection error for test_connection call
        self.mock_session.get.side_effect = requests.ConnectionError("Connection failed")

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.test_connection()

    def test_ollama_timeout_exception_when_timeout(self):
        """타임아웃 시 예외 발생 테스트."""
        # Given: Successful response for constructor, then timeout error
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_session.get.return_value = mock_response

        client = OllamaClient()

        # Now set up timeout error for test_connection call
        self.mock_session.get.side_effect = requests.Timeout("Timeout occurred")

        # When & Then: Should raise OllamaTimeoutException
        with self.assertRaises(OllamaTimeoutException):
            client.test_connection()

    def test_ollama_connection_exception_when_http_error(self):
        """HTTP 오류 시 예외 발생 테스트."""
        # Given: Successful response for constructor
        mock_success_response = Mock()
        mock_success_response.status_code = 200

        # Set up HTTP error response for test_connection call
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.raise_for_status.side_effect = requests.HTTPError("Server error")

        # Return success for constructor, then error for test_connection
        self.mock_session.get.side_effect = [mock_success_response, mock_error_response]

        client = OllamaClient()

        # When & Then: Should raise OllamaConnectionException
        with self.assertRaises(OllamaConnectionException):
            client.test_connection()


if __name__ == "__main__":
    unittest.main()
