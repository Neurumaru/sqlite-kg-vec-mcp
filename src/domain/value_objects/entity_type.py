"""
EntityType value object for representing entity classifications.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityType:
    """Immutable entity type classifier."""

    name: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("EntityType name cannot be empty")
        if not isinstance(self.name, str):
            raise ValueError("EntityType name must be a string")
        # Normalize to consistent format
        object.__setattr__(self, 'name', self.name.strip().title())

    @classmethod
    def person(cls) -> "EntityType":
        """Create Person entity type."""
        return cls("Person")

    @classmethod
    def organization(cls) -> "EntityType":
        """Create Organization entity type."""
        return cls("Organization")

    @classmethod
    def location(cls) -> "EntityType":
        """Create Location entity type."""
        return cls("Location")

    @classmethod
    def concept(cls) -> "EntityType":
        """Create Concept entity type."""
        return cls("Concept")

    @classmethod
    def product(cls) -> "EntityType":
        """Create Product entity type."""
        return cls("Product")

    @classmethod
    def event(cls) -> "EntityType":
        """Create Event entity type."""
        return cls("Event")

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
