"""
LLM adapters for language model integrations.

These adapters implement LLM service interfaces for different providers.
"""

from .ollama import OllamaClient, OllamaLLMService, OllamaKnowledgeExtractor

__all__ = [
    "OllamaClient",
    "OllamaLLMService", 
    "OllamaKnowledgeExtractor",
]
