"""
KnowledgeGraph aggregate root for the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ..value_objects.node_id import NodeId
from ..events.entity_created import EntityCreated
from ..events.relationship_created import RelationshipCreated
from ..exceptions.entity_exceptions import EntityNotFoundException, EntityAlreadyExistsException
from ..exceptions.relationship_exceptions import RelationshipNotFoundException
from .entity import Entity
from .relationship import Relationship


@dataclass
class KnowledgeGraph:
    """
    Aggregate root for the knowledge graph domain.

    This is the main entry point for all knowledge graph operations
    and ensures consistency across entities and relationships.
    """

    id: NodeId
    name: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # In-memory collections (for domain logic, not persistence)
    _entities: Dict[NodeId, Entity] = field(default_factory=dict, init=False)
    _relationships: Dict[NodeId, Relationship] = field(default_factory=dict, init=False)
    _domain_events: List = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize the knowledge graph."""
        if not self.id:
            raise ValueError("KnowledgeGraph must have an ID")

        if not self.name:
            raise ValueError("KnowledgeGraph must have a name")

    @classmethod
    def create(cls, name: str, description: Optional[str] = None) -> "KnowledgeGraph":
        """
        Factory method to create a new knowledge graph.

        Args:
            name: Name of the knowledge graph
            description: Optional description

        Returns:
            New KnowledgeGraph instance
        """
        return cls(
            id=NodeId.generate(),
            name=name,
            description=description
        )

    # Entity management
    def add_entity(self, entity: Entity) -> Entity:
        """
        Add an entity to the knowledge graph.

        Args:
            entity: Entity to add

        Returns:
            The added entity

        Raises:
            EntityAlreadyExistsException: If entity with same ID already exists
        """
        if entity.id in self._entities:
            raise EntityAlreadyExistsException(str(entity.id))

        self._entities[entity.id] = entity
        self.updated_at = datetime.now()

        # Raise domain event
        self._raise_event(EntityCreated(
            entity_id=entity.id,
            entity_type=entity.entity_type.name,
            name=entity.name,
            graph_id=self.id
        ))

        return entity

    def get_entity(self, entity_id: NodeId) -> Entity:
        """
        Get an entity by ID.

        Args:
            entity_id: ID of the entity

        Returns:
            The entity

        Raises:
            EntityNotFoundException: If entity not found
        """
        if entity_id not in self._entities:
            raise EntityNotFoundException(str(entity_id))

        return self._entities[entity_id]

    def update_entity(self, entity: Entity) -> Entity:
        """
        Update an existing entity.

        Args:
            entity: Updated entity

        Returns:
            The updated entity

        Raises:
            EntityNotFoundException: If entity not found
        """
        if entity.id not in self._entities:
            raise EntityNotFoundException(str(entity.id))

        self._entities[entity.id] = entity
        self.updated_at = datetime.now()

        return entity

    def remove_entity(self, entity_id: NodeId) -> Entity:
        """
        Remove an entity from the knowledge graph.

        Args:
            entity_id: ID of entity to remove

        Returns:
            The removed entity

        Raises:
            EntityNotFoundException: If entity not found
        """
        if entity_id not in self._entities:
            raise EntityNotFoundException(str(entity_id))

        entity = self._entities[entity_id]

        # Remove all relationships involving this entity
        related_relationships = [
            rel for rel in self._relationships.values()
            if rel.involves_entity(entity_id)
        ]

        for rel in related_relationships:
            del self._relationships[rel.id]

        del self._entities[entity_id]
        self.updated_at = datetime.now()

        return entity

    def has_entity(self, entity_id: NodeId) -> bool:
        """
        Check if entity exists in the graph.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity exists
        """
        return entity_id in self._entities

    # Relationship management
    def add_relationship(self, relationship: Relationship) -> Relationship:
        """
        Add a relationship to the knowledge graph.

        Args:
            relationship: Relationship to add

        Returns:
            The added relationship

        Raises:
            EntityNotFoundException: If source or target entity not found
        """
        # Validate that source and target entities exist
        if not self.has_entity(relationship.source_id):
            raise EntityNotFoundException(str(relationship.source_id))

        if not self.has_entity(relationship.target_id):
            raise EntityNotFoundException(str(relationship.target_id))

        self._relationships[relationship.id] = relationship
        self.updated_at = datetime.now()

        # Raise domain event
        self._raise_event(RelationshipCreated(
            relationship_id=relationship.id,
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            relationship_type=relationship.relationship_type.name,
            graph_id=self.id
        ))

        return relationship

    def get_relationship(self, relationship_id: NodeId) -> Relationship:
        """
        Get a relationship by ID.

        Args:
            relationship_id: ID of the relationship

        Returns:
            The relationship

        Raises:
            RelationshipNotFoundException: If relationship not found
        """
        if relationship_id not in self._relationships:
            raise RelationshipNotFoundException(str(relationship_id))

        return self._relationships[relationship_id]

    def update_relationship(self, relationship: Relationship) -> Relationship:
        """
        Update an existing relationship.

        Args:
            relationship: Updated relationship

        Returns:
            The updated relationship

        Raises:
            RelationshipNotFoundException: If relationship not found
        """
        if relationship.id not in self._relationships:
            raise RelationshipNotFoundException(str(relationship.id))

        self._relationships[relationship.id] = relationship
        self.updated_at = datetime.now()

        return relationship

    def remove_relationship(self, relationship_id: NodeId) -> Relationship:
        """
        Remove a relationship from the knowledge graph.

        Args:
            relationship_id: ID of relationship to remove

        Returns:
            The removed relationship

        Raises:
            RelationshipNotFoundException: If relationship not found
        """
        if relationship_id not in self._relationships:
            raise RelationshipNotFoundException(str(relationship_id))

        relationship = self._relationships[relationship_id]
        del self._relationships[relationship_id]
        self.updated_at = datetime.now()

        return relationship

    def has_relationship(self, relationship_id: NodeId) -> bool:
        """
        Check if relationship exists in the graph.

        Args:
            relationship_id: Relationship ID to check

        Returns:
            True if relationship exists
        """
        return relationship_id in self._relationships

    # Graph queries
    def get_entity_relationships(self, entity_id: NodeId) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships
        """
        return [
            rel for rel in self._relationships.values()
            if rel.involves_entity(entity_id)
        ]

    def get_neighbors(self, entity_id: NodeId) -> List[Entity]:
        """
        Get neighboring entities (connected by relationships).

        Args:
            entity_id: Entity ID

        Returns:
            List of neighboring entities
        """
        neighbor_ids = set()

        for rel in self._relationships.values():
            if rel.source_id == entity_id:
                neighbor_ids.add(rel.target_id)
            elif rel.target_id == entity_id:
                neighbor_ids.add(rel.source_id)

        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]

    def find_path(self, source_id: NodeId, target_id: NodeId, max_depth: int = 5) -> Optional[List[NodeId]]:
        """
        Find shortest path between two entities using BFS.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum search depth

        Returns:
            Path as list of entity IDs, or None if no path found
        """
        if source_id == target_id:
            return [source_id]

        if not self.has_entity(source_id) or not self.has_entity(target_id):
            return None

        from collections import deque

        queue = deque([(source_id, [source_id])])
        visited = {source_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get neighbors
            for rel in self._relationships.values():
                next_id = None
                if rel.source_id == current_id:
                    next_id = rel.target_id
                elif rel.target_id == current_id:
                    next_id = rel.source_id

                if next_id and next_id not in visited:
                    new_path = path + [next_id]

                    if next_id == target_id:
                        return new_path

                    visited.add(next_id)
                    queue.append((next_id, new_path))

        return None

    # Statistics
    @property
    def entity_count(self) -> int:
        """Get number of entities in the graph."""
        return len(self._entities)

    @property
    def relationship_count(self) -> int:
        """Get number of relationships in the graph."""
        return len(self._relationships)

    def get_entity_type_counts(self) -> Dict[str, int]:
        """
        Get count of entities by type.

        Returns:
            Dictionary mapping entity type to count
        """
        counts = {}
        for entity in self._entities.values():
            type_name = entity.entity_type.name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def get_relationship_type_counts(self) -> Dict[str, int]:
        """
        Get count of relationships by type.

        Returns:
            Dictionary mapping relationship type to count
        """
        counts = {}
        for rel in self._relationships.values():
            type_name = rel.relationship_type.name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    # Domain events
    def _raise_event(self, event):
        """Add a domain event to the event list."""
        self._domain_events.append(event)

    def get_domain_events(self) -> List:
        """Get all domain events."""
        return self._domain_events.copy()

    def clear_domain_events(self):
        """Clear all domain events."""
        self._domain_events.clear()

    def to_dict(self) -> Dict:
        """
        Convert knowledge graph to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "entity_type_counts": self.get_entity_type_counts(),
            "relationship_type_counts": self.get_relationship_type_counts(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def __str__(self) -> str:
        return f"KnowledgeGraph({self.name}, entities={self.entity_count}, relationships={self.relationship_count})"
