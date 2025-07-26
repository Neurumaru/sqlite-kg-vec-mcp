"""
Primary port for search use cases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ..domain import (
    NodeId,
    Vector,
    SearchCriteria,
    SearchResultCollection,
    EntityType,
)


class SearchUseCases(ABC):
    """
    Primary port defining search operations.

    This interface defines all the ways external systems can perform
    search operations on the knowledge graph.
    """

    # Text-based search
    @abstractmethod
    async def search_by_text(
        self,
        query_text: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> SearchResultCollection:
        """
        Search entities using text query.

        Args:
            query_text: Text query to search for
            entity_types: Optional filter by entity types
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            property_filters: Optional property filters

        Returns:
            Collection of search results
        """
        pass

    # Vector-based search
    @abstractmethod
    async def search_by_vector(
        self,
        query_vector: Vector,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> SearchResultCollection:
        """
        Search entities using vector query.

        Args:
            query_vector: Vector to search for
            entity_types: Optional filter by entity types
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            property_filters: Optional property filters

        Returns:
            Collection of search results
        """
        pass

    # Hybrid search
    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Vector,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        similarity_threshold: float = 0.0,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> SearchResultCollection:
        """
        Search entities using both text and vector queries.

        Args:
            query_text: Text query
            query_vector: Vector query
            entity_types: Optional filter by entity types
            limit: Maximum number of results to return
            text_weight: Weight for text similarity score
            vector_weight: Weight for vector similarity score
            similarity_threshold: Minimum combined similarity score
            property_filters: Optional property filters

        Returns:
            Collection of search results
        """
        pass

    # Structured search
    @abstractmethod
    async def search_by_criteria(
        self,
        criteria: SearchCriteria
    ) -> SearchResultCollection:
        """
        Search entities using structured search criteria.

        Args:
            criteria: Search criteria object

        Returns:
            Collection of search results
        """
        pass

    # Similar entities
    @abstractmethod
    async def find_similar_entities(
        self,
        entity_id: NodeId,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.5
    ) -> SearchResultCollection:
        """
        Find entities similar to a given entity.

        Args:
            entity_id: ID of the reference entity
            entity_types: Optional filter by entity types
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            Collection of similar entities
        """
        pass

    # Interactive search
    @abstractmethod
    async def interactive_search(
        self,
        query: str,
        max_steps: int = 10,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform LLM-guided interactive search.

        Args:
            query: Initial search query
            max_steps: Maximum search steps
            session_id: Optional session ID for tracking
            user_id: Optional user ID for personalization

        Returns:
            Interactive search results with metadata
        """
        pass

    # Search suggestions
    @abstractmethod
    async def get_search_suggestions(
        self,
        partial_query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 5
    ) -> List[str]:
        """
        Get search suggestions for a partial query.

        Args:
            partial_query: Partial search query
            entity_types: Optional filter by entity types
            limit: Maximum number of suggestions

        Returns:
            List of search suggestions
        """
        pass

    # Search analytics
    @abstractmethod
    async def get_search_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get search analytics and metrics.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Analytics data
        """
        pass
