"""
HNSW (Hierarchical Navigable Small World) adapters for vector similarity search.

This module contains HNSW-specific implementations for text embedding,
vector search, and HNSW indexing functionality using hnswlib and FAISS.
"""

from .embeddings import Embedding, EmbeddingManager
from .exceptions import (
    VectorException,
    EmbeddingGenerationException,
    VectorDimensionException,
    VectorSearchException,
    VectorIndexException,
    VectorStorageException,
    VectorNormalizationException,
)
from .hnsw import HNSWIndex
from .search import SearchResult, VectorSearch
from .text_embedder import VectorTextEmbedder, create_embedder

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
    "SearchResult",
    "VectorSearch",
    "VectorTextEmbedder",
    "create_embedder",
]
