"""
HNSW embeddings 어댑터의 단위 테스트.

헥사고날 아키텍처 원칙에 따라 Mock 객체를 사용하여 외부 의존성을 격리합니다.
"""

# pylint: disable=protected-access

import sqlite3
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.hnsw.embeddings import Embedding, EmbeddingManager


class TestEmbedding(unittest.TestCase):
    """Embedding 값 객체의 단위 테스트."""

    def test_from_row_with_node_type(self):
        """Given: 노드 타입의 SQLite Row 객체
        When: Embedding.from_row()를 호출하면
        Then: 적절한 Embedding 객체를 생성해야 한다
        """
        # Given
        mock_row = Mock()
        mock_row.__getitem__ = Mock(
            side_effect=lambda key: {
                "node_id": 123,
                "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
                "dimensions": 3,
                "model_info": "test_model",
                "embedding_version": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }[key]
        )

        # When
        embedding = Embedding.from_row(mock_row, "node")

        # Then
        self.assertEqual(embedding.entity_id, 123)
        self.assertEqual(embedding.entity_type, "node")
        self.assertEqual(embedding.dimensions, 3)
        self.assertEqual(embedding.model_info, "test_model")
        self.assertEqual(embedding.embedding_version, 1)
        np.testing.assert_array_equal(
            embedding.embedding, np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )

    def test_from_row_with_edge_type(self):
        """Given: 엣지 타입의 SQLite Row 객체
        When: Embedding.from_row()를 호출하면
        Then: 적절한 Embedding 객체를 생성해야 한다
        """
        # Given
        mock_row = Mock()
        mock_row.__getitem__ = Mock(
            side_effect=lambda key: {
                "edge_id": 456,
                "embedding": np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes(),
                "dimensions": 3,
                "model_info": "edge_model",
                "embedding_version": 2,
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            }[key]
        )

        # When
        embedding = Embedding.from_row(mock_row, "edge")

        # Then
        self.assertEqual(embedding.entity_id, 456)
        self.assertEqual(embedding.entity_type, "edge")
        self.assertEqual(embedding.dimensions, 3)
        self.assertEqual(embedding.model_info, "edge_model")
        self.assertEqual(embedding.embedding_version, 2)

    def test_from_row_with_invalid_entity_type(self):
        """Given: 지원하지 않는 엔티티 타입
        When: Embedding.from_row()를 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        mock_row = Mock()
        mock_row.__getitem__ = Mock(
            side_effect=lambda key: {
                "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
            }.get(key, None)
        )

        # When & Then
        with self.assertRaises(ValueError) as context:
            Embedding.from_row(mock_row, "invalid_type")

        self.assertIn("Unsupported entity type", str(context.exception))


class TestEmbeddingManager(unittest.TestCase):
    """EmbeddingManager의 단위 테스트."""

    def setUp(self):
        """테스트 환경 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        self.embedding_manager = EmbeddingManager(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결 객체
        When: EmbeddingManager를 초기화하면
        Then: 연결 객체가 저장되어야 한다
        """
        # Given & When
        manager = EmbeddingManager(self.mock_connection)

        # Then
        self.assertEqual(manager.connection, self.mock_connection)

    def test_store_embedding_new_node(self):
        """Given: 새로운 노드 임베딩
        When: store_embedding()을 호출하면
        Then: 데이터베이스에 INSERT가 실행되어야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        model_info = "test_model"
        embedding_version = 1

        # 기존 임베딩이 없음을 시뮬레이션
        self.mock_cursor.fetchone.return_value = None
        self.mock_cursor.rowcount = 1

        # When
        result = self.embedding_manager.store_embedding(
            entity_type, entity_id, embedding, model_info, embedding_version
        )

        # Then
        self.assertTrue(result)
        self.mock_cursor.execute.assert_any_call(
            "SELECT 1 FROM node_embeddings WHERE node_id = ?", (entity_id,)
        )
        # INSERT 쿼리가 호출되었는지 확인
        insert_calls = [
            call for call in self.mock_cursor.execute.call_args_list if "INSERT INTO" in str(call)
        ]
        self.assertEqual(len(insert_calls), 1)
        self.mock_connection.commit.assert_called_once()

    def test_store_embedding_update_existing(self):
        """Given: 기존 노드 임베딩
        When: store_embedding()을 호출하면
        Then: 데이터베이스에 UPDATE가 실행되어야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        embedding = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        model_info = "updated_model"

        # 기존 임베딩이 있음을 시뮬레이션
        self.mock_cursor.fetchone.return_value = (1,)
        self.mock_cursor.rowcount = 1

        # When
        result = self.embedding_manager.store_embedding(
            entity_type, entity_id, embedding, model_info
        )

        # Then
        self.assertTrue(result)
        # UPDATE 쿼리가 호출되었는지 확인
        update_calls = [
            call for call in self.mock_cursor.execute.call_args_list if "UPDATE" in str(call)
        ]
        self.assertEqual(len(update_calls), 1)

    def test_store_embedding_invalid_entity_type(self):
        """Given: 잘못된 엔티티 타입
        When: store_embedding()을 호출하면
        Then: ValueError가 발생해야 한다
        """
        # Given
        entity_type = "invalid"
        entity_id = 123
        embedding = np.array([0.1, 0.2, 0.3])
        model_info = "test_model"

        # When & Then
        with self.assertRaises(ValueError) as context:
            self.embedding_manager.store_embedding(entity_type, entity_id, embedding, model_info)

        self.assertIn("Entity type must be", str(context.exception))

    def test_get_embedding_success(self):
        """Given: 존재하는 임베딩
        When: get_embedding()을 호출하면
        Then: Embedding 객체를 반환해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123

        mock_row = Mock()
        mock_row.__getitem__ = Mock(
            side_effect=lambda key: {
                "node_id": 123,
                "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
                "dimensions": 3,
                "model_info": "test_model",
                "embedding_version": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }[key]
        )

        self.mock_cursor.fetchone.return_value = mock_row

        # When
        result = self.embedding_manager.get_embedding(entity_type, entity_id)

        # Then
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Embedding)
        self.assertEqual(result.entity_id, 123)
        self.assertEqual(result.entity_type, "node")

    def test_get_embedding_not_found(self):
        """Given: 존재하지 않는 임베딩
        When: get_embedding()을 호출하면
        Then: None을 반환해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 999
        self.mock_cursor.fetchone.return_value = None

        # When
        result = self.embedding_manager.get_embedding(entity_type, entity_id)

        # Then
        self.assertIsNone(result)

    def test_delete_embedding_success(self):
        """Given: 존재하는 임베딩
        When: delete_embedding()을 호출하면
        Then: True를 반환하고 DELETE 쿼리가 실행되어야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        self.mock_cursor.rowcount = 1

        # When
        result = self.embedding_manager.delete_embedding(entity_type, entity_id)

        # Then
        self.assertTrue(result)
        delete_calls = [
            call for call in self.mock_cursor.execute.call_args_list if "DELETE FROM" in str(call)
        ]
        self.assertEqual(len(delete_calls), 1)

    def test_delete_embedding_not_found(self):
        """Given: 존재하지 않는 임베딩
        When: delete_embedding()을 호출하면
        Then: False를 반환해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 999
        self.mock_cursor.rowcount = 0

        # When
        result = self.embedding_manager.delete_embedding(entity_type, entity_id)

        # Then
        self.assertFalse(result)

    def test_get_all_embeddings_with_filter(self):
        """Given: 특정 모델의 임베딩 요청
        When: get_all_embeddings()을 호출하면
        Then: 필터된 임베딩 목록을 반환해야 한다
        """
        # Given
        entity_type = "node"
        model_info = "test_model"
        batch_size = 10

        # 첫 번째 배치 - 데이터 있음
        mock_row1 = Mock()
        mock_row1.__getitem__ = Mock(
            side_effect=lambda key: {
                "node_id": 1,
                "embedding": np.array([0.1, 0.2], dtype=np.float32).tobytes(),
                "dimensions": 2,
                "model_info": "test_model",
                "embedding_version": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }[key]
        )

        mock_row2 = Mock()
        mock_row2.__getitem__ = Mock(
            side_effect=lambda key: {
                "node_id": 2,
                "embedding": np.array([0.3, 0.4], dtype=np.float32).tobytes(),
                "dimensions": 2,
                "model_info": "test_model",
                "embedding_version": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }[key]
        )

        # 두 번째 배치 - 데이터 없음 (루프 종료)
        self.mock_cursor.fetchall.side_effect = [
            [mock_row1, mock_row2],  # 첫 번째 배치
            [],  # 두 번째 배치 (빈 결과)
        ]

        # When
        result = self.embedding_manager.get_all_embeddings(entity_type, model_info, batch_size)

        # Then
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Embedding)
        self.assertIsInstance(result[1], Embedding)
        self.assertEqual(result[0].entity_id, 1)
        self.assertEqual(result[1].entity_id, 2)

    def test_get_outdated_embeddings(self):
        """Given: 버전이 오래된 임베딩들
        When: get_outdated_embeddings()을 호출하면
        Then: 오래된 임베딩의 ID 목록을 반환해야 한다
        """
        # Given
        entity_type = "node"
        current_version = 2
        batch_size = 10

        # 첫 번째 배치
        self.mock_cursor.fetchall.side_effect = [
            [(1,), (2,), (3,)],  # 첫 번째 배치
            [],  # 두 번째 배치 (빈 결과)
        ]

        # When
        result = self.embedding_manager.get_outdated_embeddings(
            entity_type, current_version, batch_size
        )

        # Then
        self.assertEqual(result, [1, 2, 3])

    @patch("src.adapters.hnsw.embedder_factory.create_embedder")
    def test_generate_embedding_for_entity_node(self, mock_create_embedder):
        """Given: 노드 엔티티
        When: _generate_embedding_for_entity()를 호출하면
        Then: 텍스트 기반 임베딩을 생성해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123
        model_info = "test_model"

        # 엔티티 데이터 모킹
        self.mock_cursor.fetchone.return_value = (
            "Test Entity",  # name
            "Person",  # type
            '{"age": 30, "city": "Seoul"}',  # properties
        )

        # 임베더 모킹
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_create_embedder.return_value = mock_embedder

        # When
        result = self.embedding_manager._generate_embedding_for_entity(
            entity_type, entity_id, model_info
        )

        # Then
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        mock_embedder.embed.assert_called_once()

    def test_extract_entity_text_node(self):
        """Given: 노드 엔티티
        When: _extract_entity_text()를 호출하면
        Then: 노드의 텍스트 표현을 반환해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 123

        self.mock_cursor.fetchone.return_value = (
            "John Doe",  # name
            "Person",  # type
            '{"age": 30, "occupation": "Engineer"}',  # properties
        )

        # When
        result = self.embedding_manager._extract_entity_text(entity_type, entity_id)

        # Then
        self.assertIn("Name: John Doe", result)
        self.assertIn("Type: Person", result)
        self.assertIn("age: 30", result)
        self.assertIn("occupation: Engineer", result)

    def test_extract_entity_text_edge(self):
        """Given: 엣지 엔티티
        When: _extract_entity_text()를 호출하면
        Then: 관계의 텍스트 표현을 반환해야 한다
        """
        # Given
        entity_type = "edge"
        entity_id = 456

        self.mock_cursor.fetchone.return_value = (
            "WORKS_FOR",  # relation_type
            '{"since": "2020"}',  # properties
            "John Doe",  # source_name
            "Person",  # source_type
            "Google",  # target_name
            "Company",  # target_type
        )

        # When
        result = self.embedding_manager._extract_entity_text(entity_type, entity_id)

        # Then
        self.assertIn("Relationship: WORKS_FOR", result)
        self.assertIn("From: John Doe", result)
        self.assertIn("To: Google", result)
        self.assertIn("since: 2020", result)

    def test_extract_entity_text_not_found(self):
        """Given: 존재하지 않는 엔티티
        When: _extract_entity_text()를 호출하면
        Then: 'not found' 메시지를 반환해야 한다
        """
        # Given
        entity_type = "node"
        entity_id = 999
        self.mock_cursor.fetchone.return_value = None

        # When
        result = self.embedding_manager._extract_entity_text(entity_type, entity_id)

        # Then
        self.assertIn("not found", result)

    def test_process_outbox_success(self):
        """Given: 처리할 벡터 연산들이 아웃박스에 있을 때
        When: process_outbox()를 호출하면
        Then: 연산들이 처리되고 카운트를 반환해야 한다
        """
        # Given
        batch_size = 2

        mock_operations = [
            {
                "id": 1,
                "operation_type": "insert",
                "entity_type": "node",
                "entity_id": 123,
                "model_info": "test_model",
            },
            {
                "id": 2,
                "operation_type": "delete",
                "entity_type": "node",
                "entity_id": 456,
                "model_info": None,
            },
        ]

        self.mock_cursor.fetchall.return_value = mock_operations

        # _generate_embedding_for_entity 메서드 모킹
        with (
            patch.object(self.embedding_manager, "_generate_embedding_for_entity") as mock_generate,
            patch.object(self.embedding_manager, "store_embedding") as mock_store,
            patch.object(self.embedding_manager, "delete_embedding") as mock_delete,
        ):

            mock_generate.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_store.return_value = True
            mock_delete.return_value = True

            # Mock cursor rowcount to be a number
            self.mock_cursor.rowcount = 1

            # When
            result = self.embedding_manager.process_outbox(batch_size)

            # Then
            self.assertEqual(result, 2)
            # 상태 업데이트가 호출되었는지 확인
            status_update_calls = [
                call
                for call in self.mock_cursor.execute.call_args_list
                if "UPDATE vector_outbox SET status" in str(call)
            ]
            self.assertGreater(len(status_update_calls), 0)


if __name__ == "__main__":
    unittest.main()
