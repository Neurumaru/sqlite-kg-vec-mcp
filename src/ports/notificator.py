"""
알림 서비스 포트.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotificationMessage:
    """알림 메시지."""

    recipient: str  # 수신자 (이메일, 전화번호, 사용자ID 등)
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Notificator(ABC):
    """
    알림 포트.

    다양한 채널을 통해 사용자에게 알림을 발송하는 인터페이스입니다.
    """

    @abstractmethod
    async def send_notification(self, message: NotificationMessage) -> bool:
        """
        단일 알림을 발송합니다.

        Args:
            message: 발송할 알림 메시지

        Returns:
            발송 성공 여부
        """

    @abstractmethod
    async def send_batch_notifications(self, messages: list[NotificationMessage]) -> bool:
        """
        여러 알림을 배치로 발송합니다.

        Args:
            messages: 발송할 알림 메시지 리스트

        Returns:
            배치 발송 성공 여부
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """
        알림 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
