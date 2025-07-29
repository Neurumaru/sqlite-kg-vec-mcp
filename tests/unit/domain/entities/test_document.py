"""
Document 엔티티 단위 테스트.
"""

import time
import unittest
from datetime import datetime

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocument(unittest.TestCase):
    """Document 엔티티 테스트."""

    def test_create_document_success(self):
        """문서 생성 성공 테스트."""
        # When
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="이것은 테스트 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        # Then
        self.assertEqual(document.title, "테스트 문서")
        self.assertEqual(document.content, "이것은 테스트 문서입니다.")
        self.assertEqual(document.doc_type, DocumentType.TEXT)
        self.assertEqual(document.status, DocumentStatus.PENDING)
        self.assertIsInstance(document.created_at, datetime)
        self.assertIsInstance(document.updated_at, datetime)
        self.assertIsNone(document.processed_at)
        self.assertEqual(len(document.connected_nodes), 0)
        self.assertEqual(len(document.connected_relationships), 0)
        self.assertEqual(len(document.metadata), 0)

    def test_create_document_validation_error_empty_title(self):
        """빈 제목으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="",
                content="테스트 내용",
                doc_type=DocumentType.TEXT,
            )

    def test_create_document_validation_error_whitespace_title(self):
        """공백 제목으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="   ",
                content="테스트 내용",
                doc_type=DocumentType.TEXT,
            )

    def test_create_document_validation_error_empty_content(self):
        """빈 내용으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="테스트 제목",
                content="",
                doc_type=DocumentType.TEXT,
            )

    def test_create_document_validation_error_whitespace_content(self):
        """공백 내용으로 문서 생성 시 유효성 검사 실패 테스트."""
        # When & Then
        with self.assertRaises(ValueError):
            Document(
                id=DocumentId.generate(),
                title="테스트 제목",
                content="   ",
                doc_type=DocumentType.TEXT,
            )

    def test_mark_as_processing_success(self):
        """문서 처리 중 상태 변경 성공 테스트."""

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        original_updated_at = document.updated_at

        # 시간 차이를 보장하기 위한 작은 딜레이
        time.sleep(0.001)

        # When
        document.mark_as_processing()

        # Then
        self.assertEqual(document.status, DocumentStatus.PROCESSING)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertIsNone(document.processed_at)

    def test_mark_as_processed_success(self):
        """문서 처리 완료 상태 변경 성공 테스트."""

        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        original_updated_at = document.updated_at

        # 시간 차이를 보장하기 위한 작은 딜레이
        time.sleep(0.001)

        # When
        document.mark_as_processed()

        # Then
        self.assertEqual(document.status, DocumentStatus.PROCESSED)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertIsInstance(document.processed_at, datetime)
        self.assertTrue(document.is_processed())

    def test_mark_as_failed_success(self):
        """문서 처리 실패 상태 변경 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        error_message = "처리 중 오류 발생"
        original_updated_at = document.updated_at

        # When
        document.mark_as_failed(error_message)

        # Then
        self.assertEqual(document.status, DocumentStatus.FAILED)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertEqual(document.metadata["error"], error_message)
        self.assertFalse(document.is_processed())

    def test_add_connected_node_success(self):
        """연결된 노드 추가 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        original_updated_at = document.updated_at

        # When
        document.add_connected_node(node_id)

        # Then
        self.assertIn(node_id, document.connected_nodes)
        self.assertEqual(len(document.connected_nodes), 1)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertTrue(document.has_connected_elements())

    def test_add_connected_node_duplicate_ignored(self):
        """중복된 노드 추가 시 무시되는지 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)
        original_count = len(document.connected_nodes)

        # When
        document.add_connected_node(node_id)  # 중복 추가

        # Then
        self.assertEqual(len(document.connected_nodes), original_count)
        self.assertEqual(document.connected_nodes.count(node_id), 1)

    def test_add_connected_relationship_success(self):
        """연결된 관계 추가 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        original_updated_at = document.updated_at

        # When
        document.add_connected_relationship(relationship_id)

        # Then
        self.assertIn(relationship_id, document.connected_relationships)
        self.assertEqual(len(document.connected_relationships), 1)
        self.assertGreater(document.updated_at, original_updated_at)
        self.assertTrue(document.has_connected_elements())

    def test_remove_connected_node_success(self):
        """연결된 노드 제거 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)
        original_updated_at = document.updated_at

        # When
        time.sleep(0.001)  # 타이밍 차이 보장
        document.remove_connected_node(node_id)

        # Then
        self.assertNotIn(node_id, document.connected_nodes)
        self.assertEqual(len(document.connected_nodes), 0)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_remove_connected_node_not_exists(self):
        """존재하지 않는 노드 제거 시 무시되는지 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        original_updated_at = document.updated_at
        original_count = len(document.connected_nodes)

        # When
        document.remove_connected_node(node_id)

        # Then
        self.assertEqual(len(document.connected_nodes), original_count)
        self.assertEqual(document.updated_at, original_updated_at)

    def test_remove_connected_relationship_success(self):
        """연결된 관계 제거 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        document.add_connected_relationship(relationship_id)
        original_updated_at = document.updated_at

        # When
        document.remove_connected_relationship(relationship_id)

        # Then
        self.assertNotIn(relationship_id, document.connected_relationships)
        self.assertEqual(len(document.connected_relationships), 0)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_get_word_count_success(self):
        """문서 단어 수 계산 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트",
            content="이것은 테스트 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        # When
        word_count = document.get_word_count()

        # Then
        self.assertEqual(word_count, 3)  # "이것은", "테스트", "문서입니다."

    def test_get_char_count_success(self):
        """문서 문자 수 계산 성공 테스트."""
        content = "테스트 문서"
        document = Document(
            id=DocumentId.generate(), title="테스트", content=content, doc_type=DocumentType.TEXT
        )

        # When
        char_count = document.get_char_count()

        # Then
        self.assertEqual(char_count, len(content))

    def test_update_metadata_success(self):
        """메타데이터 업데이트 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        key = "author"
        value = "홍길동"
        original_updated_at = document.updated_at

        # When
        document.update_metadata(key, value)

        # Then
        self.assertEqual(document.metadata[key], value)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_has_connected_elements_with_nodes(self):
        """노드가 연결된 경우 연결 요소 존재 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        node_id = NodeId.generate()
        document.add_connected_node(node_id)

        # When & Then
        self.assertTrue(document.has_connected_elements())

    def test_has_connected_elements_with_relationships(self):
        """관계가 연결된 경우 연결 요소 존재 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        relationship_id = RelationshipId.generate()
        document.add_connected_relationship(relationship_id)

        # When & Then
        self.assertTrue(document.has_connected_elements())

    def test_has_connected_elements_empty(self):
        """연결 요소가 없는 경우 확인 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When & Then
        self.assertFalse(document.has_connected_elements())

    def test_str_representation(self):
        """문자열 표현 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When
        str_repr = str(document)

        # Then
        self.assertIn(str(document.id), str_repr)
        self.assertIn(document.title, str_repr)
        self.assertIn(document.status.value, str_repr)

    def test_repr_representation(self):
        """repr 표현 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When
        repr_str = repr(document)

        # Then
        self.assertIn("Document", repr_str)
        self.assertIn(document.title, repr_str)
        self.assertIn(document.doc_type.value, repr_str)
        self.assertIn(document.status.value, repr_str)

    def test_increment_version_success(self):
        """버전 증가 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )
        original_version = document.version
        original_updated_at = document.updated_at

        # When
        time.sleep(0.001)  # 타이밍 차이 보장
        document.increment_version()

        # Then
        self.assertEqual(document.version, original_version + 1)
        self.assertGreater(document.updated_at, original_updated_at)

    def test_get_version_success(self):
        """버전 조회 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When
        version = document.get_version()

        # Then
        self.assertEqual(version, document.version)
        self.assertEqual(version, 1)  # 기본값

    def test_set_version_success(self):
        """버전 설정 성공 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # When
        document.set_version(5)

        # Then
        self.assertEqual(document.version, 5)

    def test_version_field_in_document_creation(self):
        """문서 생성 시 버전 필드 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
            version=3,
        )

        # Then
        self.assertEqual(document.version, 3)

    def test_version_field_default_value(self):
        """문서 생성 시 버전 필드 기본값 테스트."""
        document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT,
        )

        # Then
        self.assertEqual(document.version, 1)

    def test_concurrent_modification_scenario(self):
        """동시 수정 시나리오 시뮬레이션 테스트."""
        # Given: 동일한 문서의 두 인스턴스
        doc_id = DocumentId.generate()

        document1 = Document(
            id=doc_id, title="원본 문서", content="원본 내용", doc_type=DocumentType.TEXT, version=1
        )

        document2 = Document(
            id=doc_id, title="원본 문서", content="원본 내용", doc_type=DocumentType.TEXT, version=1
        )

        # When: 첫 번째 문서 수정
        document1.title = "수정된 문서 1"
        document1.increment_version()

        # 두 번째 문서도 수정 시도
        document2.title = "수정된 문서 2"
        document2.increment_version()

        # Then: 버전이 다름 (실제 Repository에서 충돌 감지해야 함)
        self.assertEqual(document1.version, 2)
        self.assertEqual(document2.version, 2)
        self.assertNotEqual(document1.title, document2.title)


if __name__ == "__main__":
    unittest.main()
