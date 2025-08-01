"""
HNSW (Hierarchical Navigable Small World) adapters for vector similarity search.

This module contains HNSW-specific implementations for text embedding,
vector search, and HNSW indexing functionality using hnswlib and FAISS.
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
