"""
Relationship domain model for the knowledge graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_type import RelationshipType
from src.domain.exceptions.relationship_exceptions import InvalidRelationshipException


@dataclass
class Relationship:
    """
    Rich domain entity representing a relationship between entities in the knowledge graph.

    This encapsulates business logic for relationships and ensures invariants.
    """

    id: NodeId
    source_id: NodeId
    target_id: NodeId
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate relationship invariants."""
        self._validate()

    def _validate(self):
        """Validate business rules and invariants."""
        if not self.id:
            raise InvalidRelationshipException("Relationship must have an ID")

        if not self.source_id:
            raise InvalidRelationshipException("Relationship must have a source ID")

        if not self.target_id:
            raise InvalidRelationshipException("Relationship must have a target ID")

        if not self.relationship_type:
            raise InvalidRelationshipException("Relationship must have a type")

        # Business rule: Cannot create self-referencing relationships for certain types
        if self.source_id == self.target_id:
            disallowed_self_ref = ["WORKS_FOR", "LOCATED_IN", "PART_OF"]
            if self.relationship_type.name in disallowed_self_ref:
                raise InvalidRelationshipException(
                    f"Self-referencing {self.relationship_type.name} relationships are not allowed"
                )

        # Business rule: Properties should not contain reserved keys
        reserved_keys = {"id", "source_id", "target_id", "type", "created_at", "updated_at"}
        if any(key in reserved_keys for key in self.properties.keys()):
            raise InvalidRelationshipException("Properties cannot contain reserved keys")

    @classmethod
    def create(
        cls,
        source_id: NodeId,
        target_id: NodeId,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> "Relationship":
        """
        Factory method to create a new relationship with generated ID.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relationship_type: Type of the relationship
            properties: Optional properties dictionary

        Returns:
            New relationship instance
        """
        return cls(
            id=NodeId.generate(),
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {}
        )

    @classmethod
    def restore(
        cls,
        id: NodeId,
        source_id: NodeId,
        target_id: NodeId,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ) -> "Relationship":
        """
        Restore a relationship from persistence.

        Args:
            id: Relationship ID
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of the relationship
            properties: Optional properties
            created_at: Creation timestamp
            updated_at: Last update timestamp

        Returns:
            Restored relationship instance
        """
        now = datetime.now()
        return cls(
            id=id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {},
            created_at=created_at or now,
            updated_at=updated_at or now
        )

    def update_property(self, key: str, value: Any):
        """
        Update a single property with validation.

        Args:
            key: Property key
            value: Property value
        """
        reserved_keys = {"id", "source_id", "target_id", "type", "created_at", "updated_at"}
        if key in reserved_keys:
            raise InvalidRelationshipException(f"Cannot update reserved property: {key}")

        self.properties[key] = value
        self.updated_at = datetime.now()

    def update_properties(self, new_properties: Dict[str, Any]):
        """
        Update multiple properties with validation.

        Args:
            new_properties: Dictionary of properties to update
        """
        reserved_keys = {"id", "source_id", "target_id", "type", "created_at", "updated_at"}
        invalid_keys = set(new_properties.keys()) & reserved_keys
        if invalid_keys:
            raise InvalidRelationshipException(f"Cannot update reserved properties: {invalid_keys}")

        self.properties.update(new_properties)
        self.updated_at = datetime.now()

    def remove_property(self, key: str):
        """
        Remove a property.

        Args:
            key: Property key to remove
        """
        if key in self.properties:
            del self.properties[key]
            self.updated_at = datetime.now()

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property value.

        Args:
            key: Property key
            default: Default value if key not found

        Returns:
            Property value or default
        """
        return self.properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """
        Check if relationship has a property.

        Args:
            key: Property key

        Returns:
            True if property exists
        """
        return key in self.properties

    def is_bidirectional(self) -> bool:
        """
        Check if this relationship type is typically bidirectional.

        Returns:
            True if relationship is bidirectional
        """
        bidirectional_types = ["COLLABORATES_WITH", "RELATED_TO", "CONNECTED_TO"]
        return self.relationship_type.name in bidirectional_types

    def get_reverse_type(self) -> Optional[RelationshipType]:
        """
        Get the reverse relationship type if applicable.

        Returns:
            Reverse relationship type or None
        """
        reverse_mapping = {
            "WORKS_FOR": "EMPLOYS",
            "LOCATED_IN": "CONTAINS",
            "PART_OF": "HAS_PART",
            "FOUNDED": "FOUNDED_BY",
        }

        reverse_name = reverse_mapping.get(self.relationship_type.name)
        if reverse_name:
            return RelationshipType(reverse_name)
        return None

    def involves_entity(self, entity_id: NodeId) -> bool:
        """
        Check if this relationship involves a specific entity.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity is source or target
        """
        return self.source_id == entity_id or self.target_id == entity_id

    def get_other_entity_id(self, entity_id: NodeId) -> Optional[NodeId]:
        """
        Get the other entity ID in this relationship.

        Args:
            entity_id: Known entity ID

        Returns:
            Other entity ID or None if entity not involved
        """
        if self.source_id == entity_id:
            return self.target_id
        elif self.target_id == entity_id:
            return self.source_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "target_id": str(self.target_id),
            "type": str(self.relationship_type),
            "properties": self.properties.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def __str__(self) -> str:
        return f"{self.source_id} -[{self.relationship_type}]-> {self.target_id}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Relationship):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
