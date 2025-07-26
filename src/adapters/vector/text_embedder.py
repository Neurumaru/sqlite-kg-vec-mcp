"""
Text embedding module for converting text to vectors.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TextEmbedder(ABC):
    """Abstract base class for text embedding implementations."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert batch of texts to embedding vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class SentenceTransformerEmbedder(TextEmbedder):
    """Text embedder using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.

        Args:
            model_name: Name of the model to use. Common options:
                - "all-MiniLM-L6-v2": Fast, 384-dimensional
                - "all-mpnet-base-v2": Higher quality, 768-dimensional
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual, 384-dimensional
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert batch of texts to embedding vectors."""
        result = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        # Convert to list of arrays if needed
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return [result[i] for i in range(len(result))]
        return result

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        if self._dimension is None:
            raise RuntimeError("Dimension not set")
        return self._dimension


class OpenAIEmbedder(TextEmbedder):
    """Text embedder using OpenAI's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimension: Optional[int] = None,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model: Model to use for embeddings
            dimension: For models that support it, the dimension of embeddings
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. " "Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not found")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.custom_dimension = dimension

        # Default dimensions for known models
        self._default_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self._dimension = dimension or self._default_dimensions.get(model, 1536)
        self.custom_dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        # Only pass dimensions if it's not None
        kwargs = {"input": text, "model": self.model}
        if self.custom_dimension is not None:
            kwargs["dimensions"] = self.custom_dimension
        response = self.client.embeddings.create(**kwargs)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert batch of texts to embedding vectors."""
        # Only pass dimensions if it's not None
        kwargs = {"input": texts, "model": self.model}
        if self.custom_dimension is not None:
            kwargs["dimensions"] = self.custom_dimension
        response = self.client.embeddings.create(**kwargs)
        return [np.array(data.embedding, dtype=np.float32) for data in response.data]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


class RandomEmbedder(TextEmbedder):
    """Dummy embedder that generates random vectors (for testing)."""

    def __init__(self, dimension: int = 128):
        """Initialize with specified dimension."""
        self._dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """Generate a deterministic random vector based on text."""
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # Normalize for cosine similarity
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate random vectors for batch of texts."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension


def create_embedder(
    embedder_type: str = "sentence-transformers", **kwargs
) -> TextEmbedder:
    """
    Factory function to create text embedders.

    Args:
        embedder_type: Type of embedder to create:
            - "sentence-transformers": Use sentence-transformers library
            - "openai": Use OpenAI's API
            - "nomic": Use Nomic Embed Text via Ollama
            - "random": Use random embeddings (for testing)
        **kwargs: Additional arguments passed to the embedder constructor

    Returns:
        TextEmbedder instance
    """
    if embedder_type == "sentence-transformers":
        return SentenceTransformerEmbedder(**kwargs)
    elif embedder_type == "openai":
        return OpenAIEmbedder(**kwargs)
    elif embedder_type == "nomic":
        from .nomic_embedder import NomicEmbedder
        return NomicEmbedder(**kwargs)
    elif embedder_type == "random":
        return RandomEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}. Supported: sentence-transformers, openai, nomic, random")
