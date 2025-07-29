"""
노드 관련 DTO 정의.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(Enum):
    """노드 타입."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"


@dataclass
class NodeData:
    """
    노드 데이터를 나타내는 DTO 클래스.

    Attributes:
        id: 노드의 고유 식별자
        name: 노드 이름
        node_type: 노드 타입 (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, OBJECT)
        properties: 노드의 속성 데이터
        embedding: 노드의 임베딩 벡터
        created_at: 노드 생성 시각
        updated_at: 노드 최종 업데이트 시각
        source_documents: 이 노드를 생성한 원본 문서들의 식별자 목록
        confidence_score: 노드 추출의 신뢰도 점수 (0.0-1.0)
    """

    id: str  # 노드 고유 식별자
    name: str  # 노드 이름
    node_type: NodeType  # 노드 타입
    properties: Dict[str, Any]  # 노드 속성
    embedding: Optional[List[float]] = None  # 임베딩 벡터
    created_at: Optional[datetime] = None  # 생성 시각
    updated_at: Optional[datetime] = None  # 업데이트 시각
    source_documents: List[str] = field(default_factory=list)  # 원본 문서 목록
    confidence_score: Optional[float] = None  # 신뢰도 점수

    def __post_init__(self) -> None:
        """
        객체 생성 후 데이터 검증을 수행합니다.

        Raises:
            ValueError: 잘못된 노드 데이터가 제공된 경우
            TypeError: 잘못된 타입이 제공된 경우
        """
        # id 검증
        if not isinstance(self.id, str):
            raise TypeError("id는 문자열이어야 합니다")

        if not self.id.strip():
            raise ValueError("id는 공백이 아닌 문자를 포함해야 합니다")

        # name 검증
        if not isinstance(self.name, str):
            raise TypeError("name은 문자열이어야 합니다")

        if not self.name.strip():
            raise ValueError("name은 공백이 아닌 문자를 포함해야 합니다")

        # node_type 검증
        if not isinstance(self.node_type, NodeType):
            raise TypeError("node_type은 NodeType enum이어야 합니다")

        # properties 검증
        if not isinstance(self.properties, dict):
            raise TypeError("properties는 딕셔너리여야 합니다")

        # embedding 검증
        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                raise TypeError("embedding은 리스트여야 합니다")

            if not self.embedding:
                raise ValueError("embedding이 제공된 경우 비어있을 수 없습니다")

            for i, value in enumerate(self.embedding):
                if not isinstance(value, (int, float)):
                    raise TypeError(f"embedding[{i}]는 숫자여야 합니다. 받은 타입: {type(value)}")

                # NaN이나 무한대 값 검증
                if math.isnan(value):  # NaN 체크
                    raise ValueError(f"embedding[{i}]에 NaN 값이 포함되어 있습니다")

                if abs(value) == float("inf"):
                    raise ValueError(f"embedding[{i}]에 무한대 값이 포함되어 있습니다")

        # created_at 검증
        if self.created_at is not None:
            if not isinstance(self.created_at, datetime):
                raise TypeError("created_at은 datetime 객체여야 합니다")

        # updated_at 검증
        if self.updated_at is not None:
            if not isinstance(self.updated_at, datetime):
                raise TypeError("updated_at은 datetime 객체여야 합니다")

            # created_at이 있는 경우 updated_at이 이후여야 함
            if self.created_at is not None and self.updated_at < self.created_at:
                raise ValueError("updated_at은 created_at보다 이후여야 합니다")

        # source_documents 검증
        if not isinstance(self.source_documents, list):
            raise TypeError("source_documents는 리스트여야 합니다")

        for i, doc_id in enumerate(self.source_documents):
            if not isinstance(doc_id, str):
                raise TypeError(f"source_documents[{i}]는 문자열이어야 합니다")

            if not doc_id.strip():
                raise ValueError(f"source_documents[{i}]는 공백이 아닌 문자를 포함해야 합니다")

        # confidence_score 검증
        if self.confidence_score is not None:
            if not isinstance(self.confidence_score, (int, float)):
                raise TypeError("confidence_score는 숫자여야 합니다")

            if 0.0 > self.confidence_score or self.confidence_score > 1.0:
                raise ValueError("confidence_score는 0.0과 1.0 사이의 값이어야 합니다")

            # NaN이나 무한대 값 검증
            if self.confidence_score != self.confidence_score:  # NaN 체크
                raise ValueError("confidence_score에 NaN 값이 포함되어 있습니다")

            if abs(self.confidence_score) == float("inf"):
                raise ValueError("confidence_score에 무한대 값이 포함되어 있습니다")
