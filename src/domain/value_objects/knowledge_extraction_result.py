"""
지식 추출 결과 값 객체.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship


@dataclass(frozen=True)
class KnowledgeExtractionResult:
    """
    지식 추출 결과를 나타내는 값 객체.

    문서에서 추출된 노드와 관계의 집합을 불변 형태로 보관합니다.
    """

    nodes: list[Node]
    relationships: list[Relationship]
    extracted_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """값 객체 검증."""
        if not isinstance(self.nodes, list):
            raise TypeError("nodes는 리스트여야 합니다")

        if not isinstance(self.relationships, list):
            raise TypeError("relationships는 리스트여야 합니다")

        for i, node in enumerate(self.nodes):
            if not isinstance(node, Node):
                raise TypeError(f"nodes[{i}]는 Node 인스턴스여야 합니다")

        for i, relationship in enumerate(self.relationships):
            if not isinstance(relationship, Relationship):
                raise TypeError(f"relationships[{i}]는 Relationship 인스턴스여야 합니다")

    def is_empty(self) -> bool:
        """추출된 지식이 없는지 확인."""
        return len(self.nodes) == 0 and len(self.relationships) == 0

    def get_node_count(self) -> int:
        """추출된 노드 수."""
        return len(self.nodes)

    def get_relationship_count(self) -> int:
        """추출된 관계 수."""
        return len(self.relationships)

    def get_total_count(self) -> int:
        """추출된 전체 요소 수 (노드 + 관계)."""
        return self.get_node_count() + self.get_relationship_count()

    def get_summary(self) -> dict[str, Any]:
        """추출 결과 요약 정보."""
        return {
            "node_count": self.get_node_count(),
            "relationship_count": self.get_relationship_count(),
            "total_count": self.get_total_count(),
            "extracted_at": self.extracted_at,
            "is_empty": self.is_empty(),
        }

    @classmethod
    def empty(cls) -> "KnowledgeExtractionResult":
        """빈 추출 결과 생성."""
        return cls(nodes=[], relationships=[])
