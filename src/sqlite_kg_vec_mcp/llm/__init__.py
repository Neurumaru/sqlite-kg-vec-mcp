"""
LLM integration module for SQLite KG Vec MCP.

This module provides integration with various Large Language Models
including Ollama for knowledge graph construction and text generation.
"""

from .ollama_client import OllamaClient
from .knowledge_extractor import KnowledgeExtractor

__all__ = ["OllamaClient", "KnowledgeExtractor"]