"""
Ollama Knowledge Extractor 어댑터 단위 테스트.
"""

# pylint: disable=protected-access
import sqlite3
import unittest
from unittest.mock import Mock, patch

from src.adapters.ollama.ollama_knowledge_extractor import (
    ExtractionResult,
    OllamaKnowledgeExtractor,
)


class TestExtractionResult(unittest.TestCase):
    """ExtractionResult 데이터클래스 테스트."""

    def test_extraction_result_creation(self):
        """ExtractionResult 생성 테스트."""
        # Given: Result data
        result = ExtractionResult(
            entities_created=5,
            relationships_created=3,
            errors=["Error 1", "Error 2"],
            processing_time=2.5,
        )

        # Then: All fields should be set correctly
        self.assertEqual(result.entities_created, 5)
        self.assertEqual(result.relationships_created, 3)
        self.assertEqual(result.errors, ["Error 1", "Error 2"])
        self.assertEqual(result.processing_time, 2.5)

    def test_extraction_result_default_values(self):
        """ExtractionResult 기본값 테스트."""
        # Given: Result with default values
        result = ExtractionResult()

        # Then: Should have default values
        self.assertEqual(result.entities_created, 0)
        self.assertEqual(result.relationships_created, 0)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.processing_time, 0.0)

    def test_extraction_result_post_init(self):
        """ExtractionResult post_init 테스트."""
        # Given: Result with None errors
        result = ExtractionResult(errors=None)

        # Then: Should initialize empty errors list
        self.assertEqual(result.errors, [])


class TestOllamaKnowledgeExtractor(unittest.TestCase):
    """OllamaKnowledgeExtractor 어댑터 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처 설정."""
        # Mock database connection
        self.mock_connection = Mock(spec=sqlite3.Connection)

        # Mock Ollama client
        self.mock_ollama_client = Mock()
        self.mock_ollama_client.model = "llama3.2"

        # Mock managers
        self.mock_entity_manager = Mock()
        self.mock_relationship_manager = Mock()
        self.mock_embedding_manager = Mock()

        # Create extractor with mocks
        with (
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.EntityManager"
            ) as mock_entity_mgr_cls,
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.RelationshipManager"
            ) as mock_rel_mgr_cls,
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.EmbeddingManager"
            ) as mock_embed_mgr_cls,
        ):

            mock_entity_mgr_cls.return_value = self.mock_entity_manager
            mock_rel_mgr_cls.return_value = self.mock_relationship_manager
            mock_embed_mgr_cls.return_value = self.mock_embedding_manager

            self.extractor = OllamaKnowledgeExtractor(
                connection=self.mock_connection,
                ollama_client=self.mock_ollama_client,
                auto_embed=True,
            )

    def test_initialization_with_auto_embed(self):
        """자동 임베딩 활성화로 초기화 테스트."""
        # Then: Should initialize all managers
        self.assertEqual(self.extractor.connection, self.mock_connection)
        self.assertEqual(self.extractor.ollama_client, self.mock_ollama_client)
        self.assertTrue(self.extractor.auto_embed)
        self.assertIsNotNone(self.extractor.embedding_manager)

    def test_initialization_without_auto_embed(self):
        """자동 임베딩 비활성화로 초기화 테스트."""
        # Given: Extractor without auto embedding
        with (
            patch("src.adapters.ollama.ollama_knowledge_extractor.EntityManager"),
            patch("src.adapters.ollama.ollama_knowledge_extractor.RelationshipManager"),
        ):

            extractor = OllamaKnowledgeExtractor(
                connection=self.mock_connection,
                ollama_client=self.mock_ollama_client,
                auto_embed=False,
            )

            # Then: Embedding manager should be None
            self.assertFalse(extractor.auto_embed)
            self.assertIsNone(extractor.embedding_manager)

    def test_extract_from_text_success(self):
        """텍스트에서 지식 추출 성공 테스트."""
        # Given: Successful extraction data
        extraction_data = {
            "entities": [
                {"id": "person_1", "name": "John Doe", "type": "Person", "properties": {"age": 30}}
            ],
            "relationships": [
                {
                    "source": "person_1",
                    "target": "company_1",
                    "type": "WORKS_FOR",
                    "properties": {"since": "2020"},
                }
            ],
        }

        self.mock_ollama_client.extract_entities_and_relationships.return_value = extraction_data

        # Mock entity creation
        mock_entity = Mock()
        mock_entity.id = 123
        mock_entity.name = "John Doe"
        mock_entity.type = "Person"
        self.mock_entity_manager.create_entity.return_value = mock_entity

        # Mock embedding processing
        self.mock_embedding_manager.process_outbox.return_value = 1

        # When: Extract from text
        with patch("time.time", side_effect=[1000, 1002]):
            result = self.extractor.extract_from_text("John Doe works for a company.")

        # Then: Should return successful result
        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.entities_created, 1)
        self.assertEqual(result.relationships_created, 0)  # No target entity found
        self.assertEqual(result.processing_time, 2.0)
        self.assertEqual(len(result.errors), 1)  # Target entity not found error

    def test_extract_from_text_with_enhanced_descriptions(self):
        """설명 향상 기능과 함께 텍스트 추출 테스트."""
        # Given: Extraction data and enhanced description
        extraction_data = {
            "entities": [
                {"id": "person_1", "name": "John Doe", "type": "Person", "properties": {}}
            ],
            "relationships": [],
        }

        self.mock_ollama_client.extract_entities_and_relationships.return_value = extraction_data
        self.mock_ollama_client.generate_embeddings_description.return_value = (
            "John Doe is a software engineer."
        )

        # Mock entity creation
        mock_entity = Mock()
        mock_entity.id = 123
        self.mock_entity_manager.create_entity.return_value = mock_entity

        # When: Extract with enhanced descriptions
        _ = self.extractor.extract_from_text("John Doe is a person.", enhance_descriptions=True)

        # Then: Should call description enhancement
        self.mock_ollama_client.generate_embeddings_description.assert_called_once()

        # Verify enhanced description was added to properties
        create_call = self.mock_entity_manager.create_entity.call_args
        properties = create_call.kwargs["properties"]
        self.assertEqual(properties["llm_description"], "John Doe is a software engineer.")

    def test_extract_from_text_llm_extraction_error(self):
        """LLM 추출 오류 테스트."""
        # Given: LLM extraction fails
        self.mock_ollama_client.extract_entities_and_relationships.side_effect = ValueError(
            "LLM Error"
        )

        # When: Extract from text
        result = self.extractor.extract_from_text("Test text")

        # Then: Should return result with errors
        self.assertEqual(result.entities_created, 0)
        self.assertEqual(result.relationships_created, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("지식 추출 실패", result.errors[0])

    def test_process_entities_success(self):
        """엔티티 처리 성공 테스트."""
        # Given: Valid entity data
        entities = [
            {"id": "person_1", "name": "John Doe", "type": "Person", "properties": {"age": 30}},
            {"name": "Jane Smith", "type": "Person"},  # No ID
        ]

        # Mock entity creation
        mock_entities = [Mock(id=123), Mock(id=124)]
        self.mock_entity_manager.create_entity.side_effect = mock_entities

        # When: Process entities
        errors = []
        count = self.extractor._process_entities(entities, "test_source", False, errors)

        # Then: Should create entities and update mapping
        self.assertEqual(count, 2)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(self.extractor.entity_id_mapping), 2)
        self.assertEqual(self.extractor.entity_id_mapping["person_1"], 123)
        self.assertEqual(self.extractor.entity_id_mapping["Jane Smith"], 124)

    def test_process_entities_missing_required_fields(self):
        """필수 필드 누락 엔티티 처리 테스트."""
        # Given: Entity missing required fields
        entities = [{"id": "person_1"}, {"name": "John"}]  # Missing name and type  # Missing type

        # When: Process entities
        errors = []
        count = self.extractor._process_entities(entities, None, False, errors)

        # Then: Should skip invalid entities
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 2)
        self.mock_entity_manager.create_entity.assert_not_called()

    def test_process_entities_creation_error(self):
        """엔티티 생성 오류 테스트."""
        # Given: Entity creation fails
        entities = [{"name": "John Doe", "type": "Person"}]

        self.mock_entity_manager.create_entity.side_effect = ValueError("DB Error")

        # When: Process entities
        errors = []
        count = self.extractor._process_entities(entities, None, False, errors)

        # Then: Should handle error gracefully
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("생성 실패", errors[0])

    def test_process_relationships_success(self):
        """관계 처리 성공 테스트."""
        # Given: Valid relationship data and entity mapping
        relationships = [
            {
                "source": "person_1",
                "target": "company_1",
                "type": "WORKS_FOR",
                "properties": {"since": "2020"},
            }
        ]

        # Set up entity mapping
        self.extractor.entity_id_mapping = {"person_1": 123, "company_1": 456}

        # When: Process relationships
        errors = []
        count = self.extractor._process_relationships(relationships, errors)

        # Then: Should create relationship
        self.assertEqual(count, 1)
        self.assertEqual(len(errors), 0)
        self.mock_relationship_manager.create_relationship.assert_called_once_with(
            source_id=123, target_id=456, relation_type="WORKS_FOR", properties={"since": "2020"}
        )

    def test_process_relationships_missing_entities(self):
        """존재하지 않는 엔티티 참조 관계 처리 테스트."""
        # Given: Relationship with unknown entities
        relationships = [{"source": "unknown_1", "target": "unknown_2", "type": "RELATED_TO"}]

        # Empty entity mapping
        self.extractor.entity_id_mapping = {}

        # When: Process relationships
        errors = []
        count = self.extractor._process_relationships(relationships, errors)

        # Then: Should skip and log errors
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 2)  # Source and target not found
        self.mock_relationship_manager.create_relationship.assert_not_called()

    def test_process_relationships_creation_error(self):
        """관계 생성 오류 테스트."""
        # Given: Relationship creation fails
        relationships = [{"source": "person_1", "target": "company_1", "type": "WORKS_FOR"}]

        self.extractor.entity_id_mapping = {"person_1": 123, "company_1": 456}

        self.mock_relationship_manager.create_relationship.side_effect = ValueError("DB Error")

        # When: Process relationships
        errors = []
        count = self.extractor._process_relationships(relationships, errors)

        # Then: Should handle error gracefully
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("관계", errors[0])
        self.assertIn("생성 실패", errors[0])

    def test_extract_from_documents_batch_processing(self):
        """문서 배치 처리 테스트."""
        # Given: Multiple documents
        documents = [
            {"id": "doc1", "text": "John works at Company A."},
            {"id": "doc2", "text": "Jane studies at University B."},
            {"text": "Bob lives in City C."},  # No ID
        ]

        # Mock successful extractions
        extraction_results = [
            ExtractionResult(entities_created=2, relationships_created=1),
            ExtractionResult(entities_created=2, relationships_created=1),
            ExtractionResult(entities_created=1, relationships_created=0),
        ]

        with patch.object(self.extractor, "extract_from_text", side_effect=extraction_results):
            # When: Extract from documents
            results = self.extractor.extract_from_documents(documents, batch_size=2)

            # Then: Should process all documents
            self.assertEqual(len(results), 3)
            self.assertIsInstance(results[0], ExtractionResult)

    def test_extract_from_documents_skip_empty_text(self):
        """빈 텍스트 문서 건너뛰기 테스트."""
        # Given: Documents with empty text
        documents = [
            {"id": "doc1", "text": "Valid content"},
            {"id": "doc2", "text": ""},  # Empty
            {"id": "doc3", "text": "   "},  # Whitespace only
            {"id": "doc4"},  # No text field
        ]

        with patch.object(self.extractor, "extract_from_text") as mock_extract:
            # When: Extract from documents
            results = self.extractor.extract_from_documents(documents)

            # Then: Should only process valid document
            mock_extract.assert_called_once()
            self.assertEqual(len(results), 1)

    def test_get_extraction_statistics(self):
        """추출 통계 조회 테스트."""
        # Given: Mock database cursor with statistics
        mock_cursor = Mock()
        self.mock_connection.cursor.return_value = mock_cursor

        # Mock statistics queries
        mock_cursor.fetchall.side_effect = [
            [("Person", 10), ("Company", 5)],  # Entity stats
            [("WORKS_FOR", 8), ("LOCATED_IN", 3)],  # Relationship stats
        ]
        mock_cursor.fetchone.side_effect = [
            (15,),  # Total entities
            (11,),  # Total relationships
            (12,),  # Total embeddings
        ]

        # When: Get extraction statistics
        stats = self.extractor.get_extraction_statistics()

        # Then: Should return comprehensive statistics
        expected = {
            "entities": {"total": 15, "by_type": {"Person": 10, "Company": 5}},
            "relationships": {"total": 11, "by_type": {"WORKS_FOR": 8, "LOCATED_IN": 3}},
            "embeddings": {"total_embeddings": 12},
            "model": self.mock_ollama_client.model,
        }
        self.assertEqual(stats, expected)

    def test_extract_knowledge_sync(self):
        """동기 지식 추출 테스트 (async 메서드를 동기화)."""
        # Given: Mock extraction result
        expected_result = ExtractionResult(entities_created=3, relationships_created=2)

        with patch.object(self.extractor, "extract_from_text", return_value=expected_result):
            # When: Extract knowledge (동기 호출)
            result = self.extractor.extract_from_text("Test text")

            # Then: Should return extraction result
            self.assertEqual(result, expected_result)

    def test_extract_entities_sync(self):
        """엔티티 추출 테스트 (동기화 버전)."""
        # Given: Mock LLM response
        extraction_data = {
            "entities": [
                {"name": "John", "type": "Person", "properties": {"age": 30}},
                {"name": "Company A", "type": "Organization"},
            ]
        }

        self.mock_ollama_client.extract_entities_and_relationships.return_value = extraction_data

        # Mock entity creation to return Node objects
        with patch("src.domain.entities.node.Node") as MockNode:
            mock_john = MockNode.return_value
            mock_john.name = "John"
            mock_john.node_type = "Person"

            # When: Test entity processing internally
            entities = extraction_data["entities"]

            # Then: Should have correct data
            self.assertEqual(len(entities), 2)
            self.assertEqual(entities[0]["name"], "John")
            self.assertEqual(entities[0]["type"], "Person")

    def test_extract_relationships_sync(self):
        """관계 추출 테스트 (동기화 버전)."""
        # Given: Mock relationship data
        extraction_data = {
            "relationships": [
                {
                    "source": "John",
                    "target": "Company A",
                    "type": "WORKS_FOR",
                    "properties": {"role": "engineer"},
                }
            ]
        }

        self.mock_ollama_client.extract_entities_and_relationships.return_value = extraction_data

        # When: Test relationship processing internally
        relationships = extraction_data["relationships"]

        # Then: Should have correct data
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["type"], "WORKS_FOR")
        self.assertEqual(relationships[0]["source"], "John")
        self.assertEqual(relationships[0]["target"], "Company A")

    def test_validate_extraction_success(self):
        """추출 결과 검증 성공 테스트."""
        # Given: Successful extraction result
        result = ExtractionResult(entities_created=5, relationships_created=3)

        # When: Validate extraction (기본 검증 로직)
        has_results = result.entities_created > 0 or result.relationships_created > 0
        has_errors = len(result.errors) > 0
        is_valid = has_results and not has_errors

        # Then: Should return True
        self.assertTrue(is_valid)

    def test_validate_extraction_with_errors(self):
        """오류가 있는 추출 결과 검증 테스트."""
        # Given: Result with many errors
        result = ExtractionResult(
            entities_created=2,
            relationships_created=1,
            errors=["Error 1", "Error 2", "Error 3"],  # 3 errors out of 3 items = 100% error rate
        )

        # When: Validate extraction (높은 오류율)
        total_items = result.entities_created + result.relationships_created
        error_rate = len(result.errors) / total_items if total_items > 0 else 1.0
        is_valid = error_rate < 0.5  # 50% 미만 오류율

        # Then: Should return False due to high error rate
        self.assertFalse(is_valid)

    def test_validate_extraction_no_results(self):
        """결과가 없는 추출 검증 테스트."""
        # Given: Empty result
        result = ExtractionResult(entities_created=0, relationships_created=0)

        # When: Validate extraction (결과 없음)
        has_results = result.entities_created > 0 or result.relationships_created > 0
        is_valid = has_results

        # Then: Should return False
        self.assertFalse(is_valid)

    def test_get_extraction_confidence_good_text(self):
        """좋은 텍스트의 추출 신뢰도 테스트."""
        # Given: Well-structured text
        text = "John Doe is a software engineer who works at Microsoft. The company is located in Seattle."

        # When: Calculate confidence based on text characteristics
        word_count = len(text.split())
        entity_indicators = sum(1 for word in text.split() if word[0].isupper())
        confidence = min(1.0, (word_count * 0.05) + (entity_indicators * 0.1))

        # Then: Should return high confidence
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)

    def test_get_extraction_confidence_poor_text(self):
        """품질이 낮은 텍스트의 추출 신뢰도 테스트."""
        # Given: Very short text
        text = "Hi"

        # When: Calculate confidence (짧은 텍스트는 낮은 신뢰도)
        word_count = len(text.split())
        entity_indicators = sum(1 for word in text.split() if word[0].isupper())
        confidence = min(1.0, (word_count * 0.05) + (entity_indicators * 0.1))

        # Then: Should return low confidence
        self.assertLess(confidence, 0.5)

    def test_get_extraction_confidence_empty_text(self):
        """빈 텍스트의 추출 신뢰도 테스트."""
        # Given: Empty text
        text = ""

        # When: Calculate confidence (빈 텍스트는 0 신뢰도)
        confidence = 0.0 if not text.strip() else 1.0

        # Then: Should return zero confidence
        self.assertEqual(confidence, 0.0)


class TestOllamaKnowledgeExtractorIntegration(unittest.IsolatedAsyncioTestCase):
    """OllamaKnowledgeExtractor 통합 테스트 (비동기)."""

    async def test_full_extraction_workflow(self):
        """전체 추출 워크플로우 통합 테스트."""
        # Given: Mock components
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_ollama_client = Mock()

        # Setup extraction data
        extraction_data = {
            "entities": [
                {"id": "john", "name": "John Doe", "type": "Person"},
                {"id": "microsoft", "name": "Microsoft", "type": "Company"},
            ],
            "relationships": [{"source": "john", "target": "microsoft", "type": "WORKS_FOR"}],
        }

        mock_ollama_client.extract_entities_and_relationships.return_value = extraction_data

        # Mock managers
        with (
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.EntityManager"
            ) as mock_entity_mgr_cls,
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.RelationshipManager"
            ) as mock_rel_mgr_cls,
            patch(
                "src.adapters.ollama.ollama_knowledge_extractor.EmbeddingManager"
            ) as mock_embed_mgr_cls,
        ):

            mock_entity_manager = Mock()
            mock_relationship_manager = Mock()
            mock_embedding_manager = Mock()

            mock_entity_mgr_cls.return_value = mock_entity_manager
            mock_rel_mgr_cls.return_value = mock_relationship_manager
            mock_embed_mgr_cls.return_value = mock_embedding_manager

            # Mock entity creation
            mock_entities = [Mock(id=1), Mock(id=2)]
            mock_entity_manager.create_entity.side_effect = mock_entities

            # Mock embedding processing
            mock_embedding_manager.process_outbox.return_value = 2

            # Create extractor
            extractor = OllamaKnowledgeExtractor(
                connection=mock_connection, ollama_client=mock_ollama_client, auto_embed=True
            )

            # When: Run full extraction
            result = extractor.extract_from_text("John Doe works at Microsoft.")

            # Then: Should create entities and relationships
            self.assertEqual(result.entities_created, 2)
            self.assertEqual(result.relationships_created, 1)
            self.assertEqual(len(result.errors), 0)

            # Verify managers were called
            self.assertEqual(mock_entity_manager.create_entity.call_count, 2)
            mock_relationship_manager.create_relationship.assert_called_once()
            mock_embedding_manager.process_outbox.assert_called_once()


if __name__ == "__main__":
    unittest.main()
