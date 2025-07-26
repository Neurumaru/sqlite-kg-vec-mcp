"""
Primary port for knowledge graph use cases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ..domain import (
    NodeId,
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    KnowledgeGraph,
)


class KnowledgeGraphUseCases(ABC):
    """
    Primary port defining knowledge graph operations.

    This interface defines all the ways external systems can interact
    with the knowledge graph domain.
    """

    # Knowledge Graph management
    @abstractmethod
    async def create_knowledge_graph(
        self,
        name: str,
        description: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Create a new knowledge graph.

        Args:
            name: Name of the knowledge graph
            description: Optional description

        Returns:
            Created knowledge graph
        """
        pass

    @abstractmethod
    async def get_knowledge_graph(self, graph_id: NodeId) -> KnowledgeGraph:
        """
        Get a knowledge graph by ID.

        Args:
            graph_id: ID of the knowledge graph

        Returns:
            Knowledge graph
        """
        pass

    # Entity operations
    @abstractmethod
    async def create_entity(
        self,
        entity_type: EntityType,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        Create a new entity in the knowledge graph.

        Args:
            entity_type: Type of the entity
            name: Optional name of the entity
            properties: Optional properties dictionary

        Returns:
            Created entity
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: NodeId) -> Entity:
        """
        Get an entity by ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Entity
        """
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: NodeId,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        Update an existing entity.

        Args:
            entity_id: ID of the entity to update
            name: New name (if provided)
            properties: New properties (if provided)

        Returns:
            Updated entity
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: NodeId) -> bool:
        """
        Delete an entity from the knowledge graph.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Entity]:
        """
        List entities in the knowledge graph.

        Args:
            entity_type: Optional filter by entity type
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        pass

    # Relationship operations
    @abstractmethod
    async def create_relationship(
        self,
        source_id: NodeId,
        target_id: NodeId,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> Relationship:
        """
        Create a new relationship between entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relationship_type: Type of the relationship
            properties: Optional properties dictionary

        Returns:
            Created relationship
        """
        pass

    @abstractmethod
    async def get_relationship(self, relationship_id: NodeId) -> Relationship:
        """
        Get a relationship by ID.

        Args:
            relationship_id: ID of the relationship

        Returns:
            Relationship
        """
        pass

    @abstractmethod
    async def update_relationship(
        self,
        relationship_id: NodeId,
        properties: Dict[str, Any]
    ) -> Relationship:
        """
        Update an existing relationship.

        Args:
            relationship_id: ID of the relationship to update
            properties: New properties

        Returns:
            Updated relationship
        """
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: NodeId) -> bool:
        """
        Delete a relationship from the knowledge graph.

        Args:
            relationship_id: ID of the relationship to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: NodeId,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: ID of the entity
            relationship_type: Optional filter by relationship type
            direction: Direction of relationships to include

        Returns:
            List of relationships
        """
        pass

    # Graph traversal
    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: NodeId,
        relationship_types: Optional[List[RelationshipType]] = None,
        entity_types: Optional[List[EntityType]] = None,
        direction: str = "both",
        depth: int = 1
    ) -> List[Entity]:
        """
        Get neighboring entities.

        Args:
            entity_id: ID of the central entity
            relationship_types: Optional filter by relationship types
            entity_types: Optional filter by entity types
            direction: Direction to traverse
            depth: Traversal depth

        Returns:
            List of neighboring entities
        """
        pass

    @abstractmethod
    async def find_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        max_depth: int = 5
    ) -> Optional[List[NodeId]]:
        """
        Find shortest path between two entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_depth: Maximum search depth

        Returns:
            Path as list of entity IDs, or None if no path found
        """
        pass
