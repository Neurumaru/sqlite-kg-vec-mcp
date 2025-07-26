"""
Base domain event classes.
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class DomainEvent(ABC):
    """Base class for all domain events."""
    
    def __post_init__(self):
        # Add occurred_at if not present
        if not hasattr(self, 'occurred_at'):
            object.__setattr__(self, 'occurred_at', datetime.now())

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = str(value)
            else:
                result[key] = value
        return result
