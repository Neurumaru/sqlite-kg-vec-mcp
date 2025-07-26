"""
External service ports for third-party integrations.

These interfaces define how the domain interacts with external services.
"""

from .llm_service import LLMService
from .text_embedder import TextEmbedder
from .monitoring_service import MonitoringService

__all__ = [
    "LLMService",
    "TextEmbedder",
    "MonitoringService",
]
