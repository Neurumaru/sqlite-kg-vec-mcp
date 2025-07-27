"""
벡터 값 객체.
"""

from dataclasses import dataclass
from typing import List
import math


@dataclass(frozen=True)
class Vector:
    """
    임베딩 벡터를 나타내는 값 객체.
    
    텍스트나 개념의 벡터 표현을 저장하고 
    벡터 간 유사도 계산 기능을 제공합니다.
    """
    
    values: List[float]
    
    def __post_init__(self):
        if not self.values:
            raise ValueError("Vector cannot be empty")
        if not all(isinstance(v, (int, float)) for v in self.values):
            raise ValueError("Vector values must be numeric")
    
    @property
    def dimension(self) -> int:
        """벡터의 차원."""
        return len(self.values)
    
    def magnitude(self) -> float:
        """벡터의 크기(길이) 계산."""
        return math.sqrt(sum(v * v for v in self.values))
    
    def cosine_similarity(self, other: "Vector") -> float:
        """다른 벡터와의 코사인 유사도 계산."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(self.values, other.values))
        magnitude_product = self.magnitude() * other.magnitude()
        
        if magnitude_product == 0:
            return 0.0
        
        return dot_product / magnitude_product
    
    def euclidean_distance(self, other: "Vector") -> float:
        """다른 벡터와의 유클리드 거리 계산."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.values, other.values)))
    
    def normalize(self) -> "Vector":
        """벡터 정규화."""
        magnitude = self.magnitude()
        if magnitude == 0:
            return self
        
        normalized_values = [v / magnitude for v in self.values]
        return Vector(normalized_values)
    
    def __str__(self) -> str:
        return f"Vector({self.dimension}d)"
    
    def __repr__(self) -> str:
        return f"Vector(values={self.values})"