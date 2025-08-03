"""
이벤트 관련 DTO 정의.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EventData:
    """
    도메인 이벤트 데이터를 나타내는 DTO 클래스.

    Attributes:
        event_type: 이벤트 타입 (예: "document_created", "node_updated")
        entity_id: 이벤트가 발생한 엔티티의 고유 식별자
        entity_type: 엔티티 타입 (예: "document", "node", "relationship")
        payload: 이벤트와 관련된 데이터
        timestamp: 이벤트 발생 시각
        correlation_id: 관련 이벤트들을 연결하는 식별자
        user_id: 이벤트를 발생시킨 사용자 식별자
        version: 이벤트 스키마 버전
    """

    event_type: str  # 이벤트 타입
    entity_id: str  # 엔티티 식별자
    entity_type: str  # 엔티티 타입
    payload: dict[str, Any]  # 이벤트 데이터
    timestamp: datetime  # 이벤트 발생 시각
    correlation_id: str  # 상관관계 식별자
    user_id: str | None = None  # 사용자 식별자
    version: int = 1  # 이벤트 스키마 버전

    @classmethod
    def create(
        cls,
        event_type: str,
        entity_id: str,
        entity_type: str,
        payload: dict[str, Any],
        timestamp: datetime,
        correlation_id: str | None = None,
        user_id: str | None = None,
        version: int = 1,
    ) -> "EventData":
        """
        EventData 인스턴스를 생성하는 팩토리 메서드.
        correlation_id가 None인 경우 entity_id로 설정합니다.

        인자:
            event_type: 이벤트 타입
            entity_id: 엔티티 식별자
            entity_type: 엔티티 타입
            payload: 이벤트 데이터
            timestamp: 이벤트 발생 시각
            correlation_id: 상관관계 식별자 (None인 경우 entity_id 사용)
            user_id: 사용자 식별자
            version: 이벤트 스키마 버전

        반환:
            검증된 EventData 인스턴스
        """
        final_correlation_id = correlation_id if correlation_id is not None else entity_id

        return cls(
            event_type=event_type,
            entity_id=entity_id,
            entity_type=entity_type,
            payload=payload,
            timestamp=timestamp,
            correlation_id=final_correlation_id,
            user_id=user_id,
            version=version,
        )

    def __post_init__(self) -> None:
        """
        객체 생성 후 데이터 검증 및 기본값 설정을 수행합니다.

        예외:
            ValueError: 잘못된 이벤트 데이터가 제공된 경우
            TypeError: 잘못된 타입이 제공된 경우
        """
        # event_type 검증
        if not isinstance(self.event_type, str):
            raise TypeError("event_type은 문자열이어야 합니다")

        if not self.event_type.strip():
            raise ValueError("event_type은 공백이 아닌 문자를 포함해야 합니다")

        # 유효한 이벤트 타입 패턴 검증 (snake_case)
        if not self.event_type.replace("_", "").replace("-", "").isalnum():
            raise ValueError("event_type은 영문자, 숫자, 밑줄(_), 하이픈(-)만 포함할 수 있습니다")

        # entity_id 검증
        if not isinstance(self.entity_id, str):
            raise TypeError("entity_id는 문자열이어야 합니다")

        if not self.entity_id.strip():
            raise ValueError("entity_id는 공백이 아닌 문자를 포함해야 합니다")

        # entity_type 검증
        if not isinstance(self.entity_type, str):
            raise TypeError("entity_type은 문자열이어야 합니다")

        if not self.entity_type.strip():
            raise ValueError("entity_type은 공백이 아닌 문자를 포함해야 합니다")

        # 유효한 엔티티 타입 목록 검증
        valid_entity_types = {
            "document",
            "node",
            "relationship",
            "vector",
            "embedding",
            "user",
            "session",
            "query",
            "result",
            "graph",
        }
        if self.entity_type.lower() not in valid_entity_types:
            raise ValueError(
                f"entity_type '{self.entity_type}'이 유효하지 않습니다. "
                f"유효한 타입: {', '.join(sorted(valid_entity_types))}"
            )

        # payload 검증
        if not isinstance(self.payload, dict):
            raise TypeError("payload는 딕셔너리여야 합니다")

        # timestamp 검증
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp는 datetime 객체여야 합니다")

        # 미래 시간 검증 (약간의 여유를 둠)
        now = datetime.now()
        if self.timestamp > now and (self.timestamp - now).total_seconds() > 60:
            raise ValueError("timestamp는 현재 시간보다 1분 이상 미래일 수 없습니다")

        # correlation_id 검증 (factory 메서드에서 None이 아닌 값으로 설정됨)
        if not isinstance(self.correlation_id, str):
            raise TypeError("correlation_id는 문자열이어야 합니다")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id는 공백이 아닌 문자를 포함해야 합니다")

        # user_id 검증
        if self.user_id is not None:
            if not isinstance(self.user_id, str):
                raise TypeError("user_id는 문자열이어야 합니다")

            if not self.user_id.strip():
                raise ValueError("user_id는 공백이 아닌 문자를 포함해야 합니다")

        # version 검증
        if not isinstance(self.version, int):
            raise TypeError("version은 정수여야 합니다")

        if self.version < 1:
            raise ValueError("version은 1 이상이어야 합니다")

    def generate_event_id(self) -> str:
        """
        이벤트 고유 식별자를 생성합니다.

        반환:
            UUID 기반의 이벤트 식별자
        """
        return str(uuid.uuid4())

    def is_valid_correlation(self, other_event: Any) -> bool:
        """
        다른 이벤트와의 상관관계를 확인합니다.

        인자:
            other_event: 비교할 다른 이벤트 (EventData 타입이어야 함)

        반환:
            같은 correlation_id를 가지는지 여부
        """
        if not isinstance(other_event, EventData):
            return False

        return self.correlation_id == other_event.correlation_id and self.correlation_id is not None
