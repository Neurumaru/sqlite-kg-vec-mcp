"""
MCP 작업을 위한 기본 핸들러.
"""

import logging
from typing import Any

from ..config import FastMCPConfig


class BaseHandler:
    """
    MCP 작업 핸들러의 기본 클래스.

    로깅, 응답 포맷팅과 같은 공통 기능을 제공합니다.
    """

    def __init__(self, config: FastMCPConfig):
        """
        기본 핸들러를 초기화합니다.

        Args:
            config: FastMCP 설정
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.log_level))

    def _create_success_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """표준 성공 응답을 생성합니다."""
        return {"success": True, **data}

    def _create_error_response(self, message: str, error_code: str | None = None) -> dict[str, Any]:
        """표준 오류 응답을 생성합니다."""
        return {
            "success": False,
            "error": message,
            "error_code": error_code,
        }
