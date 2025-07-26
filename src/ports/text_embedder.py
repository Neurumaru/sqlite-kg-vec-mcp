"""
Text embedder service port for text-to-vector conversion.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ..domain.value_objects.vector import Vector


class TextEmbedder(ABC):
    """
    Secondary port for text embedding operations.

    This interface defines how the domain interacts with text embedding services
    to convert text into vector representations.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> Vector:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Vector representation of the text
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of vector representations
        """
        pass

    @abstractmethod
    async def embed_with_metadata(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Vector, Dict[str, Any]]:
        """
        Generate embedding with additional metadata.

        Args:
            text: Text to embed
            metadata: Optional metadata to include

        Returns:
            Tuple of (vector, metadata) with embedding info
        """
        pass

    @abstractmethod
    async def embed_chunked_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Tuple[str, Vector]]:
        """
        Embed text by splitting into chunks.

        Args:
            text: Text to embed
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of (chunk_text, vector) tuples
        """
        pass

    @abstractmethod
    async def get_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """
        Calculate similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        pass

    @abstractmethod
    async def get_batch_similarities(
        self,
        query_vector: Vector,
        candidate_vectors: List[Vector]
    ) -> List[float]:
        """
        Calculate similarities between a query vector and multiple candidates.

        Args:
            query_vector: Query vector
            candidate_vectors: List of candidate vectors

        Returns:
            List of similarity scores
        """
        pass

    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this embedder.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    async def get_max_input_length(self) -> int:
        """
        Get the maximum input length supported by the embedder.

        Returns:
            Maximum input length in tokens
        """
        pass

    @abstractmethod
    async def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        pass

    @abstractmethod
    async def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text according to the embedder's tokenization scheme.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        pass

    @abstractmethod
    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.

        Args:
            text: Text to analyze

        Returns:
            Estimated token count
        """
        pass

    @abstractmethod
    async def supports_batch_processing(self) -> bool:
        """
        Check if the embedder supports batch processing.

        Returns:
            True if batch processing is supported
        """
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Model information including name, version, and capabilities
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the embedding service.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def warm_up(self, sample_texts: Optional[List[str]] = None) -> bool:
        """
        Warm up the embedding service with sample texts.

        Args:
            sample_texts: Optional sample texts for warming up

        Returns:
            True if warm-up was successful
        """
        pass

    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the embedding service.

        Returns:
            Usage statistics including request counts and latency metrics
        """
        pass

    @abstractmethod
    async def clear_cache(self) -> bool:
        """
        Clear any internal caches in the embedding service.

        Returns:
            True if cache clearing was successful
        """
        pass
