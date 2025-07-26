"""
Entity domain model for the knowledge graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.entity_type import EntityType
from src.domain.exceptions.entity_exceptions import InvalidEntityException


@dataclass
class Entity:
    """
    Rich domain entity representing a node in the knowledge graph.

    This is a domain entity that encapsulates business logic and invariants,
    not just a data container.
    """

    id: NodeId
    entity_type: EntityType
    name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate entity invariants."""
        self._validate()

    def _validate(self):
        """Validate business rules and invariants."""
        if not self.id:
            raise InvalidEntityException("Entity must have an ID")

        if not self.entity_type:
            raise InvalidEntityException("Entity must have a type")

        # Business rule: Name should be provided for certain entity types
        if self.entity_type.name in ["Person", "Organization", "Product"] and not self.name:
            raise InvalidEntityException(f"{self.entity_type.name} entities must have a name")

        # Business rule: Properties should not contain reserved keys
        reserved_keys = {"id", "type", "name", "created_at", "updated_at"}
        if any(key in reserved_keys for key in self.properties.keys()):
            raise InvalidEntityException("Properties cannot contain reserved keys")

    @classmethod
    def create(
        cls,
        entity_type: EntityType,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> "Entity":
        """
        Factory method to create a new entity with generated ID.

        Args:
            entity_type: Type of the entity
            name: Optional name of the entity
            properties: Optional properties dictionary

        Returns:
            New entity instance
        """
        return cls(
            id=NodeId.generate(),
            entity_type=entity_type,
            name=name,
            properties=properties or {}
        )

    @classmethod
    def restore(
        cls,
        id: NodeId,
        entity_type: EntityType,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ) -> "Entity":
        """
        Restore an entity from persistence (without validation of creation rules).

        Args:
            id: Entity ID
            entity_type: Type of the entity
            name: Optional name
            properties: Optional properties
            created_at: Creation timestamp
            updated_at: Last update timestamp

        Returns:
            Restored entity instance
        """
        now = datetime.now()
        return cls(
            id=id,
            entity_type=entity_type,
            name=name,
            properties=properties or {},
            created_at=created_at or now,
            updated_at=updated_at or now
        )

    def update_name(self, new_name: str):
        """
        Update entity name with business rule validation.

        Args:
            new_name: New name for the entity
        """
        if not new_name and self.entity_type.name in ["Person", "Organization", "Product"]:
            raise InvalidEntityException(f"{self.entity_type.name} entities must have a name")

        self.name = new_name
        self.updated_at = datetime.now()

    def update_property(self, key: str, value: Any):
        """
        Update a single property with validation.

        Args:
            key: Property key
            value: Property value
        """
        reserved_keys = {"id", "type", "name", "created_at", "updated_at"}
        if key in reserved_keys:
            raise InvalidEntityException(f"Cannot update reserved property: {key}")

        self.properties[key] = value
        self.updated_at = datetime.now()

    def update_properties(self, new_properties: Dict[str, Any]):
        """
        Update multiple properties with validation.

        Args:
            new_properties: Dictionary of properties to update
        """
        reserved_keys = {"id", "type", "name", "created_at", "updated_at"}
        invalid_keys = set(new_properties.keys()) & reserved_keys
        if invalid_keys:
            raise InvalidEntityException(f"Cannot update reserved properties: {invalid_keys}")

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
        Check if entity has a property.

        Args:
            key: Property key

        Returns:
            True if property exists
        """
        return key in self.properties

    def is_same_type(self, other: "Entity") -> bool:
        """
        Check if this entity has the same type as another.

        Args:
            other: Other entity to compare

        Returns:
            True if types match
        """
        return self.entity_type == other.entity_type

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "type": str(self.entity_type),
            "name": self.name,
            "properties": self.properties.copy(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def __str__(self) -> str:
        name_part = f" '{self.name}'" if self.name else ""
        return f"{self.entity_type}({self.id}){name_part}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
