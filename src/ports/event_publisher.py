"""
이벤트 퍼블리셔 포트.
"""

from abc import ABC, abstractmethod

from src.dto import EventData


class EventPublisher(ABC):
    """
    이벤트 발행 포트.

    도메인 이벤트를 외부 시스템에 발행하는 인터페이스입니다.
    """

    @abstractmethod
    async def publish(self, event: EventData) -> None:
        """
        단일 이벤트를 발행합니다.

        Args:
            event: 발행할 이벤트 데이터
        """

    @abstractmethod
    async def publish_batch(self, events: list[EventData]) -> None:
        """
        여러 이벤트를 배치로 발행합니다.

        Args:
            events: 발행할 이벤트 데이터 리스트
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """
        이벤트 퍼블리셔가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
