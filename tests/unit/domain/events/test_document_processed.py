"""
DocumentProcessed 이벤트 단위 테스트.
"""

import unittest
import uuid
from datetime import datetime

from src.domain.events.base import DomainEvent
from src.domain.events.document_processed import DocumentProcessed


class TestDocumentProcessed(unittest.TestCase):
    """DocumentProcessed 이벤트 테스트."""

    def test_create_document_processed_event_success(self):
        """DocumentProcessed 이벤트 생성 성공 테스트."""
        # When
        document_id = "doc-123"
        extracted_node_count = 5
        extracted_relationship_count = 3
        processing_time_seconds = 2.5

        event = DocumentProcessed.create(
            aggregate_id=document_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds,
        )

        # Then
        self.assertIsInstance(event, DocumentProcessed)
        self.assertEqual(event.document_id, document_id)
        self.assertEqual(event.aggregate_id, document_id)
        self.assertEqual(event.extracted_node_count, extracted_node_count)
        self.assertEqual(event.extracted_relationship_count, extracted_relationship_count)
        self.assertEqual(event.processing_time_seconds, processing_time_seconds)
        self.assertEqual(event.event_type, "DocumentProcessed")
        self.assertEqual(event.version, 1)
        self.assertIsInstance(event.event_id, str)
        self.assertIsInstance(event.occurred_at, datetime)
        self.assertIsInstance(event.metadata, dict)

    def test_create_document_processed_event_with_zero_extraction(self):
        """추출된 노드와 관계가 없는 경우 이벤트 생성 테스트."""
        # When
        document_id = "doc-empty"
        extracted_node_count = 0
        extracted_relationship_count = 0
        processing_time_seconds = 1.0

        event = DocumentProcessed.create(
            aggregate_id=document_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds,
        )

        # Then
        self.assertEqual(event.extracted_node_count, 0)
        self.assertEqual(event.extracted_relationship_count, 0)
        self.assertEqual(event.processing_time_seconds, 1.0)
        self.assertEqual(event.document_id, document_id)

    def test_create_document_processed_event_with_manual_construction(self):
        """수동 생성자를 통한 DocumentProcessed 이벤트 생성 테스트."""
        # When
        event_id = str(uuid.uuid4())
        occurred_at = datetime.now()
        document_id = "doc-manual"
        extracted_node_count = 10
        extracted_relationship_count = 7
        processing_time_seconds = 5.2

        event = DocumentProcessed(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type="DocumentProcessed",
            aggregate_id=document_id,
            document_id=document_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds,
        )

        # Then
        self.assertEqual(event.event_id, event_id)
        self.assertEqual(event.occurred_at, occurred_at)
        self.assertEqual(event.document_id, document_id)
        self.assertEqual(event.extracted_node_count, extracted_node_count)
        self.assertEqual(event.extracted_relationship_count, extracted_relationship_count)
        self.assertEqual(event.processing_time_seconds, processing_time_seconds)

    def test_document_processed_event_with_custom_metadata(self):
        """사용자 정의 메타데이터와 함께 이벤트 생성 테스트."""
        # When
        event_id = str(uuid.uuid4())
        occurred_at = datetime.now()
        document_id = "doc-metadata"
        metadata = {"extractor_version": "1.2.3", "model_used": "gpt-4", "source": "api_upload"}

        event = DocumentProcessed(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type="DocumentProcessed",
            aggregate_id=document_id,
            document_id=document_id,
            extracted_node_count=3,
            extracted_relationship_count=2,
            processing_time_seconds=1.8,
            metadata=metadata,
        )

        # Then
        self.assertEqual(event.metadata, metadata)
        self.assertEqual(event.metadata["extractor_version"], "1.2.3")
        self.assertEqual(event.metadata["model_used"], "gpt-4")
        self.assertEqual(event.metadata["source"], "api_upload")

    def test_document_processed_event_with_large_numbers(self):
        """큰 수치들로 이벤트 생성 테스트."""
        # When
        document_id = "doc-large"
        extracted_node_count = 1000
        extracted_relationship_count = 5000
        processing_time_seconds = 120.75

        event = DocumentProcessed.create(
            aggregate_id=document_id,
            extracted_node_count=extracted_node_count,
            extracted_relationship_count=extracted_relationship_count,
            processing_time_seconds=processing_time_seconds,
        )

        # Then
        self.assertEqual(event.extracted_node_count, 1000)
        self.assertEqual(event.extracted_relationship_count, 5000)
        self.assertEqual(event.processing_time_seconds, 120.75)

    def test_document_processed_event_inheritance(self):
        """DocumentProcessed가 DomainEvent를 올바르게 상속하는지 테스트."""
        # When
        event = DocumentProcessed.create(
            aggregate_id="doc-inheritance",
            extracted_node_count=1,
            extracted_relationship_count=1,
            processing_time_seconds=1.0,
        )

        # Then
        self.assertIsInstance(event, DomainEvent)
        self.assertTrue(hasattr(event, "event_id"))
        self.assertTrue(hasattr(event, "occurred_at"))
        self.assertTrue(hasattr(event, "event_type"))
        self.assertTrue(hasattr(event, "aggregate_id"))
        self.assertTrue(hasattr(event, "version"))
        self.assertTrue(hasattr(event, "metadata"))

    def test_multiple_document_processed_events_unique(self):
        """여러 DocumentProcessed 이벤트가 고유한지 테스트."""
        # When
        event1 = DocumentProcessed.create(
            aggregate_id="doc-1",
            extracted_node_count=1,
            extracted_relationship_count=1,
            processing_time_seconds=1.0,
        )

        event2 = DocumentProcessed.create(
            aggregate_id="doc-2",
            extracted_node_count=2,
            extracted_relationship_count=2,
            processing_time_seconds=2.0,
        )

        # Then
        self.assertNotEqual(event1.event_id, event2.event_id)
        self.assertNotEqual(event1.document_id, event2.document_id)
        self.assertNotEqual(event1.extracted_node_count, event2.extracted_node_count)


if __name__ == "__main__":
    unittest.main()
