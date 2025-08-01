"""
최적화된 문서 저장소 포트.

실제 사용 현황 분석을 바탕으로 불필요한 메서드를 제거하고
핵심 기능에 집중한 간소화된 인터페이스입니다.
"""

from abc import ABC, abstractmethod

from src.dto import DocumentData, DocumentStatus


class DocumentRepository(ABC):
    """
    최적화된 문서 저장소 포트.

    실제 사용 패턴을 분석하여 필수 기능만 남긴 경량화된 인터페이스.
    과도한 추상화를 제거하고 현재 요구사항에 집중합니다.
    """

    # 핵심 CRUD 작업 - 실제로 가장 많이 사용되는 메서드들
    @abstractmethod
    async def save(self, document: DocumentData) -> DocumentData:
        """
        문서를 저장합니다.

        Args:
            document: 저장할 문서

        Returns:
            저장된 문서
        """

    @abstractmethod
    async def find_by_id(self, document_id: str) -> DocumentData | None:
        """
        ID로 문서를 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            찾은 문서 또는 None
        """

    @abstractmethod
    async def update(self, document: DocumentData) -> DocumentData:
        """
        문서를 업데이트합니다.

        Args:
            document: 업데이트할 문서

        Returns:
            업데이트된 문서
        """

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """
        문서를 삭제합니다.

        Args:
            document_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """

    @abstractmethod
    async def exists(self, document_id: str) -> bool:
        """
        문서가 존재하는지 확인합니다.

        Args:
            document_id: 확인할 문서 ID

        Returns:
            존재 여부
        """

    # 실제 사용되는 특수 기능들
    @abstractmethod
    async def find_by_status(self, status: DocumentStatus) -> list[DocumentData]:
        """
        상태로 문서를 찾습니다.

        Args:
            status: 문서 상태

        Returns:
            해당 상태의 문서들
        """

    @abstractmethod
    async def find_unprocessed(self, limit: int = 100) -> list[DocumentData]:
        """
        처리되지 않은 문서들을 찾습니다.

        Args:
            limit: 최대 반환 개수

        Returns:
            미처리 문서들
        """

    @abstractmethod
    async def update_with_knowledge(
        self,
        document: DocumentData,
        node_ids: list[str],
        relationship_ids: list[str],
    ) -> DocumentData:
        """
        문서를 지식 요소들과 함께 업데이트합니다.

        Args:
            document: 업데이트할 문서
            node_ids: 연결된 노드 ID들
            relationship_ids: 연결된 관계 ID들

        Returns:
            업데이트된 문서
        """
