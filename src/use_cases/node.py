"""
노드 관련 Use Cases 포트.
"""

from abc import ABC, abstractmethod

from src.domain.entities.node import Node, NodeType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.vector import Vector


class NodeManagementUseCase(ABC):
    """노드 관리 Use Case 인터페이스."""

    @abstractmethod
    async def create_node(
        self,
        name: str,
        node_type: NodeType,
        description: str | None = None,
        properties: dict | None = None,
        source_documents: list[DocumentId] | None = None,
    ) -> Node:
        """새 노드를 생성합니다."""

    @abstractmethod
    async def get_node(self, node_id: NodeId) -> Node | None:
        """노드를 조회합니다."""

    @abstractmethod
    async def list_nodes(
        self,
        node_type: NodeType | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Node]:
        """노드 목록을 조회합니다."""

    @abstractmethod
    async def update_node(
        self,
        node_id: NodeId,
        name: str | None = None,
        description: str | None = None,
        properties: dict | None = None,
    ) -> Node:
        """노드를 업데이트합니다."""

    @abstractmethod
    async def delete_node(self, node_id: NodeId) -> bool:
        """노드를 삭제합니다."""

    @abstractmethod
    async def add_node_to_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        """노드를 문서에 연결합니다."""

    @abstractmethod
    async def remove_node_from_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        """노드의 문서 연결을 해제합니다."""


class NodeEmbeddingUseCase(ABC):
    """노드 임베딩 관련 Use Case 인터페이스."""

    @abstractmethod
    async def generate_node_embedding(self, node_id: NodeId) -> Vector:
        """노드의 임베딩을 생성합니다."""

    @abstractmethod
    async def update_node_embedding(self, node_id: NodeId, embedding: Vector) -> bool:
        """노드의 임베딩을 업데이트합니다."""

    @abstractmethod
    async def get_node_embedding(self, node_id: NodeId) -> Vector | None:
        """노드의 임베딩을 조회합니다."""

    @abstractmethod
    async def find_similar_nodes(
        self, node_id: NodeId, limit: int = 10, threshold: float = 0.7
    ) -> list[tuple[Node, float]]:
        """유사한 노드들을 찾습니다."""

    @abstractmethod
    async def batch_generate_embeddings(self, node_ids: list[NodeId]) -> dict[NodeId, Vector]:
        """
        노드들의 임베딩을 일괄 생성합니다.

        인자:
            node_ids: 임베딩을 생성할 노드 ID 목록 (비어있지 않아야 함)

        반환:
            노드 ID별 임베딩 벡터 딕셔너리

        예외:
            ValueError: node_ids가 비어있는 경우
        """

    @abstractmethod
    async def batch_create_nodes(
        self, node_data: list[tuple[str, NodeType, str | None, dict | None]]
    ) -> list[Node]:
        """
        여러 노드를 일괄 생성합니다.

        인자:
            node_data: (name, type, description, properties) 튜플들의 목록

        반환:
            생성된 노드 목록

        예외:
            ValueError: 입력 데이터가 잘못된 경우
        """
