"""
Repository port for relationship persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ....domain import NodeId, Relationship, RelationshipType


class RelationshipRepository(ABC):
    """
    Secondary port for relationship persistence operations.

    This interface defines how the domain interacts with relationship storage.
    """

    @abstractmethod
    async def save(self, relationship: Relationship) -> Relationship:
        """
        Save a relationship to persistent storage.

        Args:
            relationship: Relationship to save

        Returns:
            Saved relationship with updated metadata
        """
        pass

    @abstractmethod
    async def find_by_id(self, relationship_id: NodeId) -> Optional[Relationship]:
        """
        Find a relationship by its ID.

        Args:
            relationship_id: ID of the relationship to find

        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_entities(
        self,
        source_id: Optional[NodeId] = None,
        target_id: Optional[NodeId] = None,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Relationship]:
        """
        Find relationships by entity IDs.

        Args:
            source_id: Optional source entity ID
            target_id: Optional target entity ID
            relationship_type: Optional filter by relationship type
            direction: Direction to search

        Returns:
            List of matching relationships
        """
        pass

    @abstractmethod
    async def find_by_type(
        self,
        relationship_type: RelationshipType,
        limit: int = 100,
        offset: int = 0
    ) -> List[Relationship]:
        """
        Find relationships by type.

        Args:
            relationship_type: Type of relationships to find
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def find_by_properties(
        self,
        property_filters: Dict[str, Any],
        relationship_type: Optional[RelationshipType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Relationship]:
        """
        Find relationships by property values.

        Args:
            property_filters: Property filters to match
            relationship_type: Optional filter by relationship type
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching relationships
        """
        pass

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: NodeId,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "both"
    ) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: ID of the entity
            relationship_type: Optional filter by relationship type
            direction: Direction of relationships

        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def find_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        max_depth: int = 5,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> Optional[List[Relationship]]:
        """
        Find shortest path between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum search depth
            relationship_types: Optional filter by relationship types

        Returns:
            Path as list of relationships, or None if no path found
        """
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: NodeId,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[NodeId]:
        """
        Get neighboring entity IDs.

        Args:
            entity_id: Central entity ID
            relationship_types: Optional filter by relationship types
            direction: Direction to traverse
            depth: Traversal depth

        Returns:
            List of neighboring entity IDs
        """
        pass

    @abstractmethod
    async def update(self, relationship: Relationship) -> Relationship:
        """
        Update an existing relationship.

        Args:
            relationship: Relationship with updated data

        Returns:
            Updated relationship
        """
        pass

    @abstractmethod
    async def delete(self, relationship_id: NodeId) -> bool:
        """
        Delete a relationship from storage.

        Args:
            relationship_id: ID of the relationship to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def delete_by_entity(self, entity_id: NodeId) -> int:
        """
        Delete all relationships connected to an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            Number of relationships deleted
        """
        pass

    @abstractmethod
    async def exists(self, relationship_id: NodeId) -> bool:
        """
        Check if a relationship exists.

        Args:
            relationship_id: ID of the relationship to check

        Returns:
            True if relationship exists
        """
        pass

    @abstractmethod
    async def count(
        self,
        relationship_type: Optional[RelationshipType] = None,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count relationships matching criteria.

        Args:
            relationship_type: Optional filter by relationship type
            property_filters: Optional property filters

        Returns:
            Number of matching relationships
        """
        pass

    @abstractmethod
    async def bulk_save(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Save multiple relationships in batch.

        Args:
            relationships: List of relationships to save

        Returns:
            List of saved relationships
        """
        pass

    @abstractmethod
    async def bulk_delete(self, relationship_ids: List[NodeId]) -> int:
        """
        Delete multiple relationships in batch.

        Args:
            relationship_ids: List of relationship IDs to delete

        Returns:
            Number of relationships deleted
        """
        pass
