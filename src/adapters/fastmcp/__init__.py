"""
지식 그래프 API를 위한 FastMCP 서버 어댑터.

이 모듈은 MCP(Model Context Protocol)를 통해
지식 그래프를 제공하기 위한 FastMCP 관련 구현을 포함합니다.
"""

from .refactored_server import KnowledgeGraphServer

__all__ = ["KnowledgeGraphServer"]
