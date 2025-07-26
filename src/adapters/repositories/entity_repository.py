"""
Repository port for entity persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.domain import NodeId, Entity, EntityType


class EntityRepository(ABC):
    """
    Secondary port for entity persistence operations.

    This interface defines how the domain interacts with entity storage.
    """

    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """
        Save an entity to persistent storage.

        Args:
            entity: Entity to save

        Returns:
            Saved entity with updated metadata
        """
        pass

    @abstractmethod
    async def find_by_id(self, entity_id: NodeId) -> Optional[Entity]:
        """
        Find an entity by its ID.

        Args:
            entity_id: ID of the entity to find

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """
        Find entities by name.

        Args:
            name: Name to search for
            entity_type: Optional filter by entity type

        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    async def find_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
        offset: int = 0
    ) -> List[Entity]:
        """
        Find entities by type.

        Args:
            entity_type: Type of entities to find
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of entities
        """
        pass

    @abstractmethod
    async def find_by_properties(
        self,
        property_filters: Dict[str, Any],
        entity_type: Optional[EntityType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Entity]:
        """
        Find entities by property values.

        Args:
            property_filters: Property filters to match
            entity_type: Optional filter by entity type
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching entities
        """
        pass

    @abstractmethod
    async def update(self, entity: Entity) -> Entity:
        """
        Update an existing entity.

        Args:
            entity: Entity with updated data

        Returns:
            Updated entity
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: NodeId) -> bool:
        """
        Delete an entity from storage.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: NodeId) -> bool:
        """
        Check if an entity exists.

        Args:
            entity_id: ID of the entity to check

        Returns:
            True if entity exists
        """
        pass

    @abstractmethod
    async def count(
        self,
        entity_type: Optional[EntityType] = None,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count entities matching criteria.

        Args:
            entity_type: Optional filter by entity type
            property_filters: Optional property filters

        Returns:
            Number of matching entities
        """
        pass

    @abstractmethod
    async def bulk_save(self, entities: List[Entity]) -> List[Entity]:
        """
        Save multiple entities in batch.

        Args:
            entities: List of entities to save

        Returns:
            List of saved entities
        """
        pass

    @abstractmethod
    async def bulk_delete(self, entity_ids: List[NodeId]) -> int:
        """
        Delete multiple entities in batch.

        Args:
            entity_ids: List of entity IDs to delete

        Returns:
            Number of entities deleted
        """
        pass
