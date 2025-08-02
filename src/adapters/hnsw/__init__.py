"""
벡터 유사도 검색을 위한 HNSW(Hierarchical Navigable Small World) 어댑터.

이 모듈은 hnswlib와 FAISS를 사용하여 텍스트 임베딩, 벡터 검색 및
HNSW 인덱싱 기능을 위한 HNSW 관련 구현을 포함합니다.
"""

from .embeddings import Embedding, EmbeddingManager
from .exceptions import (
    EmbeddingGenerationException,
    VectorDimensionException,
    VectorException,
    VectorIndexException,
    VectorNormalizationException,
    VectorSearchException,
    VectorStorageException,
)
from .hnsw import HNSWIndex

# from .search import SearchResult, VectorSearch  # TODO: Fix search dependencies
# from .text_embedder import VectorTextEmbedder, create_embedder  # TODO: Implement text_embedder module

__all__ = [
    "Embedding",
    "EmbeddingManager",
    "VectorException",
    "EmbeddingGenerationException",
    "VectorDimensionException",
    "VectorSearchException",
    "VectorIndexException",
    "VectorStorageException",
    "VectorNormalizationException",
    "HNSWIndex",
    # "SearchResult",  # TODO: Re-enable when dependencies are fixed
    # "VectorSearch",  # TODO: Re-enable when dependencies are fixed
    # "VectorTextEmbedder",  # TODO: Re-enable when text_embedder is implemented
    # "create_embedder",  # TODO: Re-enable when text_embedder is implemented
]
