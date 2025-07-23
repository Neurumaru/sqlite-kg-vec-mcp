"""
NodeId value object for representing unique entity identifiers.
"""

import uuid
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class NodeId:
    """Immutable identifier for graph nodes/entities."""

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("NodeId value cannot be empty")
        if not isinstance(self.value, str):
            raise ValueError("NodeId value must be a string")

    @classmethod
    def generate(cls) -> "NodeId":
        """Generate a new UUID-based NodeId."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_int(cls, int_id: int) -> "NodeId":
        """Create NodeId from integer (for backward compatibility)."""
        return cls(str(int_id))

    @classmethod
    def from_uuid(cls, uuid_str: str) -> "NodeId":
        """Create NodeId from UUID string."""
        # Validate UUID format
        try:
            uuid.UUID(uuid_str)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {uuid_str}")
        return cls(uuid_str)

    def to_int(self) -> Union[int, None]:
        """Convert to integer if possible (for backward compatibility)."""
        try:
            return int(self.value)
        except ValueError:
            return None

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)
