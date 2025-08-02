"""
매퍼 포트 인터페이스 정의.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship


class DocumentMapper(ABC):
    """문서 엔티티와 DTO 간 매핑을 위한 추상 인터페이스."""

    @abstractmethod
    def to_data(self, document: Document) -> Any:
        """도메인 엔티티를 DTO로 변환."""

    @abstractmethod
    def from_data(self, data: Any) -> Document:
        """DTO를 도메인 엔티티로 변환."""


class NodeMapper(ABC):
    """노드 엔티티와 DTO 간 매핑을 위한 추상 인터페이스."""

    @abstractmethod
    def to_data(self, node: Node) -> Any:
        """도메인 엔티티를 DTO로 변환."""

    @abstractmethod
    def from_data(self, data: Any) -> Node:
        """DTO를 도메인 엔티티로 변환."""


class RelationshipMapper(ABC):
    """관계 엔티티와 DTO 간 매핑을 위한 추상 인터페이스."""

    @abstractmethod
    def to_data(self, relationship: Relationship) -> Any:
        """도메인 엔티티를 DTO로 변환."""

    @abstractmethod
    def from_data(self, data: Any) -> Relationship:
        """DTO를 도메인 엔티티로 변환."""
