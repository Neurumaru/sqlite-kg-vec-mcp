"""
RelationshipType value object for representing relationship classifications.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationshipType:
    """Immutable relationship type classifier."""

    name: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("RelationshipType name cannot be empty")
        if not isinstance(self.name, str):
            raise ValueError("RelationshipType name must be a string")
        # Normalize to consistent format (uppercase with underscores)
        object.__setattr__(self, 'name', self.name.strip().upper().replace(' ', '_'))

    @classmethod
    def works_for(cls) -> "RelationshipType":
        """Create WORKS_FOR relationship type."""
        return cls("WORKS_FOR")

    @classmethod
    def located_in(cls) -> "RelationshipType":
        """Create LOCATED_IN relationship type."""
        return cls("LOCATED_IN")

    @classmethod
    def founded(cls) -> "RelationshipType":
        """Create FOUNDED relationship type."""
        return cls("FOUNDED")

    @classmethod
    def collaborates_with(cls) -> "RelationshipType":
        """Create COLLABORATES_WITH relationship type."""
        return cls("COLLABORATES_WITH")

    @classmethod
    def part_of(cls) -> "RelationshipType":
        """Create PART_OF relationship type."""
        return cls("PART_OF")

    @classmethod
    def related_to(cls) -> "RelationshipType":
        """Create RELATED_TO relationship type."""
        return cls("RELATED_TO")

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
