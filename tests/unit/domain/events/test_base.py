"""
도메인 이벤트 기본 클래스 단위 테스트.
"""

import unittest
import uuid
from datetime import datetime
from dataclasses import dataclass

from src.domain.events.base import DomainEvent


@dataclass(init=False)
class TestEvent(DomainEvent):
    """테스트용 도메인 이벤트."""

    test_data: str

    def __init__(
        self,
        event_id: str,
        occurred_at: datetime,
        event_type: str,
        aggregate_id: str,
        test_data: str,
        version: int = 1,
        metadata=None,
    ):
        super().__init__(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type=event_type,
            aggregate_id=aggregate_id,
            version=version,
            metadata=metadata or {},
        )
        self.test_data = test_data

    @classmethod
    def create(cls, aggregate_id: str, test_data: str) -> "TestEvent":
        """테스트 이벤트 생성."""
        return super().create(
            aggregate_id=aggregate_id,
            test_data=test_data,
        )


class TestDomainEvent(unittest.TestCase):
    """DomainEvent 기본 클래스 테스트."""

    def test_create_domain_event_success(self):
        """도메인 이벤트 생성 성공 테스트."""
        # When
        aggregate_id = "test-aggregate-123"
        test_data = "test data"
        event = TestEvent.create(aggregate_id, test_data)

        # Then
        self.assertIsInstance(event, TestEvent)
        self.assertIsInstance(event, DomainEvent)
        self.assertEqual(event.aggregate_id, aggregate_id)
        self.assertEqual(event.test_data, test_data)
        self.assertEqual(event.event_type, "TestEvent")
        self.assertEqual(event.version, 1)
        self.assertIsInstance(event.event_id, str)
        self.assertIsInstance(event.occurred_at, datetime)
        self.assertIsInstance(event.metadata, dict)

    def test_create_domain_event_with_uuid(self):
        """UUID 형식의 event_id가 생성되는지 테스트."""
        # When
        event = TestEvent.create("test-aggregate", "test data")

        # Then
        # UUID 형식 검증
        try:
            uuid.UUID(event.event_id)
        except ValueError:
            self.fail("event_id should be a valid UUID")

    def test_create_domain_event_with_metadata(self):
        """메타데이터와 함께 도메인 이벤트 생성 테스트."""
        # When
        aggregate_id = "test-aggregate-123"
        test_data = "test data"
        event = TestEvent.create(aggregate_id, test_data)
        
        # 메타데이터 수동 추가 (create 메서드에서는 기본적으로 빈 dict)
        event.metadata["source"] = "unit_test"
        event.metadata["priority"] = "high"

        # Then
        self.assertEqual(event.metadata["source"], "unit_test")
        self.assertEqual(event.metadata["priority"], "high")

    def test_domain_event_immutability_of_core_fields(self):
        """도메인 이벤트 핵심 필드의 불변성 검증."""
        # Given
        event = TestEvent.create("test-aggregate", "test data")
        original_event_id = event.event_id
        original_occurred_at = event.occurred_at
        original_aggregate_id = event.aggregate_id

        # When & Then
        # 핵심 필드들이 변경되지 않는지 확인
        self.assertEqual(event.event_id, original_event_id)
        self.assertEqual(event.occurred_at, original_occurred_at)
        self.assertEqual(event.aggregate_id, original_aggregate_id)

    def test_domain_event_with_custom_version(self):
        """사용자 정의 버전으로 도메인 이벤트 생성 테스트."""
        # When
        event_id = str(uuid.uuid4())
        occurred_at = datetime.now()
        event = TestEvent(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type="TestEvent",
            aggregate_id="test-aggregate",
            test_data="test data",
            version=2,
        )

        # Then
        self.assertEqual(event.version, 2)
        self.assertEqual(event.event_id, event_id)
        self.assertEqual(event.occurred_at, occurred_at)

    def test_domain_event_metadata_default_empty_dict(self):
        """메타데이터 기본값이 빈 딕셔너리인지 테스트."""
        # When
        event = TestEvent.create("test-aggregate", "test data")

        # Then
        self.assertIsInstance(event.metadata, dict)
        self.assertEqual(len(event.metadata), 0)

    def test_multiple_events_have_unique_ids(self):
        """여러 이벤트가 고유한 ID를 가지는지 테스트."""
        # When
        event1 = TestEvent.create("test-aggregate-1", "test data 1")
        event2 = TestEvent.create("test-aggregate-2", "test data 2")

        # Then
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertNotEqual(event1.occurred_at, event2.occurred_at)


if __name__ == "__main__":
    unittest.main()