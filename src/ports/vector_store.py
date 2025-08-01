"""
Vector store port with LangChain compatibility.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


class VectorStore(ABC):
    """
    Vector store port with LangChain compatibility.

    Provides vector storage and search functionality compatible with LangChain's VectorStore interface.
    """

    # Core LangChain VectorStore methods
    @abstractmethod
    async def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """
        Add documents to the vector store (LangChain standard).

        Args:
            documents: Document objects to add
            **kwargs: Additional options

        Returns:
            List of IDs of added documents
        """

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Perform similarity search (LangChain standard).

        Args:
            query: Search query string
            k: Number of documents to return
            **kwargs: Additional search options (filter, etc.)

        Returns:
            List of similar Document objects
        """

    @abstractmethod
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return with scores (LangChain standard).

        Args:
            query: Search query string
            k: Number of documents to return
            **kwargs: Additional search options

        Returns:
            List of (Document, similarity_score) tuples
        """

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Perform similarity search by vector (LangChain standard).

        Args:
            embedding: Query vector (embedding)
            k: Number of documents to return
            **kwargs: Additional search options

        Returns:
            List of similar Document objects
        """

    @abstractmethod
    async def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """
        Delete documents (LangChain standard).

        Args:
            ids: Document IDs to delete
            **kwargs: Additional deletion options

        Returns:
            Success status of deletion
        """

    # Class methods (LangChain pattern)
    @classmethod
    @abstractmethod
    async def from_documents(
        cls,
        documents: list[Document],
        embedding: Any,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        Create vector store from documents (LangChain standard).

        Args:
            documents: Initial Document objects
            embedding: Embedding model
            **kwargs: Additional creation options

        Returns:
            Created VectorStore instance
        """

    @classmethod
    @abstractmethod
    async def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """
        Create vector store from texts (LangChain standard).

        Args:
            texts: List of texts
            embedding: Embedding model
            metadatas: Metadata for each text
            **kwargs: Additional creation options

        Returns:
            Created VectorStore instance
        """

    # Additional helper methods
    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Convert vector store to Retriever (LangChain standard).

        Args:
            **kwargs: Retriever configuration options

        Returns:
            Retriever instance
        """
