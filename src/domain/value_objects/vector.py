"""
Vector value object for representing embedding vectors.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Vector:
    """Immutable vector for embeddings."""

    values: tuple[float, ...]

    def __post_init__(self):
        if not self.values:
            raise ValueError("Vector cannot be empty")
        if not all(isinstance(v, (int, float)) for v in self.values):
            raise ValueError("All vector values must be numeric")

    @classmethod
    def from_list(cls, values: List[float]) -> "Vector":
        """Create vector from list of floats."""
        return cls(tuple(values))

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Vector":
        """Create vector from numpy array."""
        if array.ndim != 1:
            raise ValueError("Array must be 1-dimensional")
        return cls(tuple(array.astype(float).tolist()))

    @classmethod
    def zeros(cls, dimension: int) -> "Vector":
        """Create zero vector of given dimension."""
        return cls(tuple(0.0 for _ in range(dimension)))

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return len(self.values)

    def to_list(self) -> List[float]:
        """Convert to list."""
        return list(self.values)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values, dtype=float)

    def dot(self, other: "Vector") -> float:
        """Compute dot product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        return sum(a * b for a, b in zip(self.values, other.values))

    def norm(self) -> float:
        """Compute L2 norm of vector."""
        return np.sqrt(sum(v * v for v in self.values))

    def cosine_similarity(self, other: "Vector") -> float:
        """Compute cosine similarity with another vector."""
        dot_product = self.dot(other)
        norms = self.norm() * other.norm()
        if norms == 0:
            return 0.0
        return dot_product / norms

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]
