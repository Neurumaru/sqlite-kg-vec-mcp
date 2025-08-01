"""
Factory module for creating text embedders.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class VectorTextEmbedder(ABC):
    """Synchronous text embedder interface for HNSW search."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""


class SyncRandomTextEmbedder(VectorTextEmbedder):
    """Random text embedder for testing purposes."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Generate random embedding."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def create_embedder(embedder_type: str, **kwargs) -> VectorTextEmbedder:
    """
    Factory function to create text embedders.

    Args:
        embedder_type: Type of embedder to create
        **kwargs: Additional arguments for embedder creation

    Returns:
        VectorTextEmbedder instance
    """
    if embedder_type == "random":
        dimension = kwargs.get("dimension", 128)
        return SyncRandomTextEmbedder(dimension=dimension)
    if embedder_type == "sentence-transformers":
        # Fallback to random for now
        dimension = kwargs.get("dimension", 384)
        return SyncRandomTextEmbedder(dimension=dimension)
    raise ValueError(f"Unknown embedder type: {embedder_type}")
