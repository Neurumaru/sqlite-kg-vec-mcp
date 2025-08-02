"""
Ollama Client 테스트용 공통 기반 클래스.
"""

from unittest.mock import Mock, patch


class BaseOllamaClientTestCase:
    """OllamaClient 테스트용 공통 기반 클래스."""

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
