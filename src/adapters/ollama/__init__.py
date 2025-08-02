"""
Ollama LLM 어댑터 구현.

이 모듈은 클라이언트, 지식 추출기, LLM 서비스, Nomic 임베더를 포함한
Ollama 관련 LLM 서비스 구현을 모두 포함합니다.
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
