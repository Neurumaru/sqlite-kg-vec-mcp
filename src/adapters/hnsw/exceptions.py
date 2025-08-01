"""
Vector processing infrastructure exceptions.

These exceptions handle vector operations, embeddings generation,
and vector database errors.
"""

from typing import Any

from ..exceptions.base import InfrastructureException
from ..exceptions.data import DataValidationException


class VectorException(InfrastructureException):
    """
    Base exception for vector processing errors.

    Covers issues with vector operations, indexing,
    and vector data management.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        vector_dimension: int | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector exception.

        Args:
            operation: Vector operation being performed
            message: Detailed error message
            vector_dimension: Vector dimension if relevant
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.operation = operation
        self.vector_dimension = vector_dimension

        full_message = f"Vector {operation} failed: {message}"
        if vector_dimension:
            full_message += f" (dimension: {vector_dimension})"

        super().__init__(
            message=full_message,
            error_code=error_code or "VECTOR_ERROR",
            context=context,
            original_error=original_error,
        )


class EmbeddingGenerationException(VectorException):
    """
    Embedding generation failures.

    Handles errors during text-to-vector conversion,
    model loading, and embedding computation.
    """

    def __init__(
        self,
        text: str,
        model_name: str,
        message: str,
        expected_dimension: int | None = None,
        actual_dimension: int | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize embedding generation exception.

        Args:
            text: Input text (truncated for logging)
            model_name: Embedding model used
            message: Detailed error message
            expected_dimension: Expected vector dimension
            actual_dimension: Actual vector dimension received
            context: Additional context
            original_error: Original exception
        """
        self.text = text[:200] + "..." if len(text) > 200 else text
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension

        full_message = f"Embedding generation failed for model '{model_name}': {message}"
        if expected_dimension and actual_dimension:
            full_message += f" (expected dim: {expected_dimension}, got: {actual_dimension})"

        super().__init__(
            operation="embedding generation",
            message=full_message,
            vector_dimension=expected_dimension,
            error_code="EMBEDDING_GENERATION_FAILED",
            context=context,
            original_error=original_error,
        )


class VectorDimensionException(DataValidationException):
    """
    Vector dimension mismatches.

    Handles cases where vector dimensions don't match
    expected values or are incompatible.
    """

    def __init__(
        self,
        expected_dimension: int,
        actual_dimension: int,
        operation: str,
        vector_id: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector dimension exception.

        Args:
            expected_dimension: Expected vector dimension
            actual_dimension: Actual vector dimension
            operation: Operation being performed
            vector_id: Vector identifier if available
            context: Additional context
            original_error: Original exception
        """
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension
        self.operation = operation
        self.vector_id = vector_id

        message = f"Vector dimension mismatch in {operation}: expected {expected_dimension}, got {actual_dimension}"
        if vector_id:
            message += f" (vector: {vector_id})"

        super().__init__(
            field="vector_dimension",
            value=actual_dimension,
            expected_format=f"{expected_dimension}D vector",
            message=message,
            error_code="VECTOR_DIMENSION_MISMATCH",
            context=context,
            original_error=original_error,
        )


class VectorSearchException(VectorException):
    """
    Vector search and similarity errors.

    Handles failures during vector search, similarity computation,
    and index querying.
    """

    def __init__(
        self,
        query_vector_dimension: int,
        index_dimension: int | None = None,
        search_params: dict[str, Any] | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector search exception.

        Args:
            query_vector_dimension: Query vector dimension
            index_dimension: Index vector dimension
            search_params: Search parameters used
            message: Optional custom message
            context: Additional context
            original_error: Original exception
        """
        self.query_vector_dimension = query_vector_dimension
        self.index_dimension = index_dimension
        self.search_params = search_params or {}

        if message is None:
            message = "Vector search failed"
            if index_dimension and query_vector_dimension != index_dimension:
                message += f": dimension mismatch (query: {query_vector_dimension}, index: {index_dimension})"

        super().__init__(
            operation="vector search",
            message=message,
            vector_dimension=query_vector_dimension,
            error_code="VECTOR_SEARCH_FAILED",
            context=context,
            original_error=original_error,
        )


class VectorIndexException(VectorException):
    """
    Vector index errors.

    Handles issues with vector index creation, updates,
    and maintenance operations.
    """

    def __init__(
        self,
        index_name: str,
        operation: str,
        message: str,
        vector_count: int | None = None,
        dimension: int | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector index exception.

        Args:
            index_name: Name of the vector index
            operation: Index operation being performed
            message: Detailed error message
            vector_count: Number of vectors in index
            dimension: Vector dimension
            context: Additional context
            original_error: Original exception
        """
        self.index_name = index_name
        self.vector_count = vector_count

        full_message = f"Vector index '{index_name}' {operation} failed: {message}"
        if vector_count:
            full_message += f" (vectors: {vector_count})"

        super().__init__(
            operation=f"index {operation}",
            message=full_message,
            vector_dimension=dimension,
            error_code="VECTOR_INDEX_ERROR",
            context=context,
            original_error=original_error,
        )


class VectorStorageException(VectorException):
    """
    Vector storage and persistence errors.

    Handles failures in storing, retrieving, and managing
    vector data in storage systems.
    """

    def __init__(
        self,
        storage_type: str,
        operation: str,
        entity_id: str | None = None,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector storage exception.

        Args:
            storage_type: Type of storage (SQLite, file, etc.)
            operation: Storage operation
            entity_id: Entity identifier if relevant
            message: Optional custom message
            context: Additional context
            original_error: Original exception
        """
        self.storage_type = storage_type
        self.entity_id = entity_id

        if message is None:
            message = f"Vector storage operation '{operation}' failed in {storage_type}"
            if entity_id:
                message += f" for entity {entity_id}"

        super().__init__(
            operation=f"storage {operation}",
            message=message,
            error_code="VECTOR_STORAGE_ERROR",
            context=context,
            original_error=original_error,
        )


class VectorNormalizationException(VectorException):
    """
    Vector normalization errors.

    Handles issues with vector normalization, validation,
    and preprocessing operations.
    """

    def __init__(
        self,
        vector_shape: tuple,
        normalization_type: str,
        message: str,
        vector_stats: dict[str, float] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize vector normalization exception.

        Args:
            vector_shape: Shape of the problematic vector
            normalization_type: Type of normalization attempted
            message: Detailed error message
            vector_stats: Vector statistics (mean, std, etc.)
            context: Additional context
            original_error: Original exception
        """
        self.vector_shape = vector_shape
        self.normalization_type = normalization_type
        self.vector_stats = vector_stats or {}

        full_message = f"Vector normalization ({normalization_type}) failed for shape {vector_shape}: {message}"

        super().__init__(
            operation=f"normalization ({normalization_type})",
            message=full_message,
            vector_dimension=vector_shape[0] if vector_shape else None,
            error_code="VECTOR_NORMALIZATION_FAILED",
            context=context,
            original_error=original_error,
        )
