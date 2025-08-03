"""
문서 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentStatus(Enum):
    """문서 상태."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(Enum):
    """문서 타입."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class DocumentData:
    """
    문서 데이터를 나타내는 DTO 클래스.

    Attributes:
        id: 문서의 고유 식별자
        title: 문서 제목
        content: 문서 내용
        doc_type: 문서 타입 (TEXT, PDF, HTML, MARKDOWN)
        status: 문서 처리 상태 (PENDING, PROCESSING, COMPLETED, FAILED)
        metadata: 문서와 관련된 추가 메타데이터
        version: 문서 버전 번호
        created_at: 문서 생성 시각
        updated_at: 문서 최종 업데이트 시각
        processed_at: 문서 처리 완료 시각
        connected_nodes: 연결된 노드 식별자 목록
        connected_relationships: 연결된 관계 식별자 목록
    """

    id: str  # 문서 고유 식별자
    title: str  # 문서 제목
    content: str  # 문서 내용
    doc_type: DocumentType  # 문서 타입
    status: DocumentStatus  # 처리 상태
    metadata: dict[str, Any]
    version: int  # 버전 번호
    created_at: datetime  # 생성 시각
    updated_at: datetime  # 업데이트 시각
    processed_at: datetime | None = None  # 처리 완료 시각
    connected_nodes: list[str] = field(default_factory=list)  # 연결된 노드 목록
    connected_relationships: list[str] = field(default_factory=list)  # 연결된 관계 목록

    def __post_init__(self) -> None:
        """
        객체 생성 후 데이터 검증을 수행합니다.

        Raises:
            ValueError: 잘못된 문서 데이터가 제공된 경우
            TypeError: 잘못된 타입이 제공된 경우
        """
        # id 검증
        if not isinstance(self.id, str):
            raise TypeError("id는 문자열이어야 합니다")

        if not self.id.strip():
            raise ValueError("id는 공백이 아닌 문자를 포함해야 합니다")

        # title 검증
        if not isinstance(self.title, str):
            raise TypeError("title은 문자열이어야 합니다")

        if not self.title.strip():
            raise ValueError("title은 공백이 아닌 문자를 포함해야 합니다")

        # content 검증
        if not isinstance(self.content, str):
            raise TypeError("content는 문자열이어야 합니다")

        # doc_type 검증
        if not isinstance(self.doc_type, DocumentType):
            raise TypeError("doc_type은 DocumentType enum이어야 합니다")

        # status 검증
        if not isinstance(self.status, DocumentStatus):
            raise TypeError("status는 DocumentStatus enum이어야 합니다")

        # metadata 검증
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata는 딕셔너리여야 합니다")

        # version 검증
        if not isinstance(self.version, int):
            raise TypeError("version은 정수여야 합니다")

        if self.version < 1:
            raise ValueError("version은 1 이상이어야 합니다")

        # created_at 검증
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at은 datetime 객체여야 합니다")

        # updated_at 검증
        if not isinstance(self.updated_at, datetime):
            raise TypeError("updated_at은 datetime 객체여야 합니다")

        # updated_at이 created_at보다 이후여야 함
        if self.updated_at < self.created_at:
            raise ValueError("updated_at은 created_at보다 이후여야 합니다")

        # processed_at 검증
        if self.processed_at is not None:
            if not isinstance(self.processed_at, datetime):
                raise TypeError("processed_at은 datetime 객체여야 합니다")

            if self.processed_at < self.created_at:
                raise ValueError("processed_at은 created_at보다 이후여야 합니다")

        # connected_nodes 검증
        if not isinstance(self.connected_nodes, list):
            raise TypeError("connected_nodes는 리스트여야 합니다")

        for i, node_id in enumerate(self.connected_nodes):
            if not isinstance(node_id, str):
                raise TypeError(f"connected_nodes[{i}]는 문자열이어야 합니다")

            if not node_id.strip():
                raise ValueError(f"connected_nodes[{i}]는 공백이 아닌 문자를 포함해야 합니다")

        # connected_relationships 검증
        if not isinstance(self.connected_relationships, list):
            raise TypeError("connected_relationships는 리스트여야 합니다")

        for i, rel_id in enumerate(self.connected_relationships):
            if not isinstance(rel_id, str):
                raise TypeError(f"connected_relationships[{i}]는 문자열이어야 합니다")

            if not rel_id.strip():
                raise ValueError(
                    f"connected_relationships[{i}]는 공백이 아닌 문자를 포함해야 합니다"
                )
