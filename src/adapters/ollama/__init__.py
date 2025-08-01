"""
Ollama LLM adapter implementations.

This module contains all Ollama-specific implementations for LLM services
including the client, knowledge extractor, LLM service, and Nomic embedder.
"""

# TODO: Fix NomicEmbedder import - missing hnsw.text_embedder dependency
# from .nomic_embedder import NomicEmbedder
from .ollama_client import OllamaClient
from .ollama_knowledge_extractor import OllamaKnowledgeExtractor
from .ollama_llm_service import OllamaLLMService

__all__ = [
    "OllamaClient",
    "OllamaKnowledgeExtractor",
    "OllamaLLMService",
    # "NomicEmbedder",  # TODO: Re-enable when hnsw.text_embedder is fixed
]
