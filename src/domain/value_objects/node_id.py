"""
노드 식별자 값 객체.
"""

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeId:
    """
    노드의 고유 식별자.

    지식 그래프의 노드를 유일하게 식별하는 값 객체입니다.
    """

    value: str

    def __post_init__(self):
        if not self.value:
            raise ValueError("NodeId cannot be empty")
        if not isinstance(self.value, str):
            raise ValueError("NodeId must be a string")

    @classmethod
    def generate(cls) -> "NodeId":
        """새로운 노드 ID 생성."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "NodeId":
        """문자열로부터 노드 ID 생성."""
        return cls(value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"NodeId('{self.value}')"
