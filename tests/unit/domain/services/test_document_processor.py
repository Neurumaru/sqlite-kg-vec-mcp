"""
DocumentProcessor 도메인 서비스 단위 테스트.
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.document_processor import DocumentProcessor, KnowledgeExtractionResult
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


class TestDocumentProcessor(unittest.TestCase):
    """DocumentProcessor 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        self.mock_knowledge_extractor = Mock()
        self.processor = DocumentProcessor(self.mock_knowledge_extractor)
        
        self.document = Document(
            id=DocumentId.generate(),
            title="테스트 문서",
            content="테스트 내용",
            doc_type=DocumentType.TEXT
        )
        
        self.sample_node = Node(
            id=NodeId.generate(),
            name="테스트 노드",
            node_type=NodeType.PERSON
        )
        
        self.sample_relationship = Relationship(
            id=RelationshipId.generate(),
            source_node_id=NodeId.generate(),
            target_node_id=NodeId.generate(),
            relationship_type=RelationshipType.WORKS_AT,
            label="근무"
        )
        
    async def test_process_document_success(self):
        """문서 처리 성공 테스트."""
        # Mock 설정
        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            return_value=([self.sample_node], [self.sample_relationship])
        )
        
        # 실행
        result = await self.processor.process_document(self.document)
        
        # 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(self.document.status, DocumentStatus.PROCESSED)
        self.assertIn(self.sample_node.id, self.document.connected_nodes)
        
    async def test_process_document_failure(self):
        """문서 처리 실패 테스트."""
        # Mock 설정 - 예외 발생
        self.mock_knowledge_extractor.extract_knowledge = AsyncMock(
            side_effect=Exception("추출 실패")
        )
        
        # 실행 및 검증
        with self.assertRaises(Exception):
            await self.processor.process_document(self.document)
            
        self.assertEqual(self.document.status, DocumentStatus.FAILED)
        self.assertEqual(self.document.metadata["error"], "추출 실패")
        
    def test_validate_document_for_processing(self):
        """문서 처리 가능성 검증 테스트."""
        # 유효한 문서
        valid_document = Document(
            id=DocumentId.generate(),
            title="유효한 문서",
            content="내용이 있는 문서",
            doc_type=DocumentType.TEXT
        )
        self.assertTrue(self.processor.validate_document_for_processing(valid_document))
        
        # 이미 처리 중인 문서
        processing_document = Document(
            id=DocumentId.generate(),
            title="처리 중인 문서",
            content="내용",
            doc_type=DocumentType.TEXT
        )
        processing_document.mark_as_processing()
        self.assertFalse(self.processor.validate_document_for_processing(processing_document))
        
        # 내용이 없는 문서는 생성자에서 예외가 발생하므로 try-except로 처리
        try:
            empty_document = Document(
                id=DocumentId.generate(),
                title="빈 문서",
                content="",
                doc_type=DocumentType.TEXT
            )
            # 여기 도달하면 안됨
            self.fail("빈 content로 Document 생성이 허용되면 안됨")
        except ValueError:
            # 예상된 동작 - 빈 content는 Document 생성 시 예외 발생
            pass


class TestKnowledgeExtractionResult(unittest.TestCase):
    """KnowledgeExtractionResult 테스트 케이스."""
    
    def test_empty_result(self):
        """빈 결과 테스트."""
        result = KnowledgeExtractionResult([], [])
        
        self.assertTrue(result.is_empty())
        self.assertEqual(result.get_node_count(), 0)
        self.assertEqual(result.get_relationship_count(), 0)
        
    def test_non_empty_result(self):
        """비어있지 않은 결과 테스트."""
        node = Node(
            id=NodeId.generate(),
            name="테스트 노드",
            node_type=NodeType.PERSON
        )
        relationship = Relationship(
            id=RelationshipId.generate(),
            source_node_id=NodeId.generate(),
            target_node_id=NodeId.generate(),
            relationship_type=RelationshipType.WORKS_AT,
            label="근무"
        )
        
        result = KnowledgeExtractionResult([node], [relationship])
        
        self.assertFalse(result.is_empty())
        self.assertEqual(result.get_node_count(), 1)
        self.assertEqual(result.get_relationship_count(), 1)


if __name__ == "__main__":
    unittest.main()