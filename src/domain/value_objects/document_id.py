"""
문서 식별자 값 객체.
"""

from dataclasses import dataclass
from typing import Union
import uuid


@dataclass(frozen=True)
class DocumentId:
    """
    문서의 고유 식별자.
    
    문서를 유일하게 식별하는 값 객체입니다.
    """
    
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("DocumentId cannot be empty")
        if not isinstance(self.value, str):
            raise ValueError("DocumentId must be a string")
    
    @classmethod
    def generate(cls) -> "DocumentId":
        """새로운 문서 ID 생성."""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> "DocumentId":
        """문자열로부터 문서 ID 생성."""
        return cls(value)
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"DocumentId('{self.value}')"