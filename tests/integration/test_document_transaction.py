"""
Document 트랜잭션 통합 테스트.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.exceptions.document_exceptions import (
    ConcurrentModificationError,
    DocumentAlreadyExistsException,
)
from src.domain.services.document_processor import (
    DocumentProcessor,
    KnowledgeExtractionResult,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto import NodeData
from src.dto import NodeType as DTONodeType
from src.dto import RelationshipData
from src.dto import RelationshipType as DTORelationshipType


class TestDocumentTransactionIntegration(unittest.IsolatedAsyncioTestCase):
    """Document 트랜잭션 통합 테스트."""

    def setUp(self):
        """테스트 설정."""
        # Mock 데이터베이스
        self.mock_database = Mock()

        # Mock transaction context manager
        transaction_mock = MagicMock()
        transaction_mock.__aenter__ = AsyncMock()
        transaction_mock.__aexit__ = AsyncMock(return_value=None)
        self.mock_database.transaction.return_value = transaction_mock

        # Mock 지식 추출기
        self.mock_knowledge_extractor = AsyncMock()

        # Repository와 Processor 생성
        self.document_repository = SQLiteDocumentRepository(self.mock_database)
        self.document_processor = DocumentProcessor(
            self.mock_knowledge_extractor, self.document_repository
        )

        # 샘플 데이터
        self.sample_document = Document(
            id=DocumentId.generate(),
            title="통합 테스트 문서",
            content="이것은 통합 테스트용 문서입니다.",
            doc_type=DocumentType.TEXT,
        )

        self.sample_node_data = NodeData(
            id=str(NodeId.generate()),
            name="테스트 개체",
            node_type=DTONodeType.CONCEPT,
            properties={"description": "테스트용 개체"},
        )

        self.sample_relationship_data = RelationshipData(
            id=str(RelationshipId.generate()),
            source_node_id=str(NodeId.generate()),
            target_node_id=str(NodeId.generate()),
            relationship_type=DTORelationshipType.CONTAINS,
            properties={"label": "포함", "strength": 0.8},
        )

    async def test_successful_document_processing_with_persistence(self):
        """영속성을 포함한 문서 처리 성공 시나리오 테스트."""
        # Given: 지식 추출이 성공적으로 수행됨
        self.mock_knowledge_extractor.extract = AsyncMock(
            return_value=([self.sample_node_data], [self.sample_relationship_data])
        )

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서가 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)  # 명령 성공

        # When: 문서를 처리
        result = await self.document_processor.process_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.relationships), 1)
        # Compare entity values instead of direct object comparison
        self.assertEqual(result.nodes[0].name, "테스트 개체")
        self.assertEqual(result.relationships[0].properties.get("label"), "포함")

        # 문서 상태가 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

        # 연결된 요소들이 문서에 추가됨
        self.assertEqual(len(self.sample_document.connected_nodes), 1)
        self.assertEqual(len(self.sample_document.connected_relationships), 1)

        # 데이터베이스 호출 검증
        self.mock_database.execute_command.assert_called()  # save 및 update 호출됨

    async def test_failed_document_processing_with_rollback(self):
        """문서 처리 실패 시 롤백 시나리오 테스트."""
        # Given: 지식 추출이 실패함
        error_message = "지식 추출 실패"
        self.mock_knowledge_extractor.extract = AsyncMock(side_effect=Exception(error_message))

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서가 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)  # 명령 성공

        # When & Then: 문서 처리가 실패하고 예외가 발생
        with self.assertRaises(Exception) as context:
            await self.document_processor.process_document(self.sample_document)

        self.assertEqual(str(context.exception), error_message)

        # 문서 상태가 FAILED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.FAILED)
        self.assertEqual(self.sample_document.metadata["error"], error_message)

        # 실패 상태 업데이트를 위한 데이터베이스 호출 검증
        self.mock_database.execute_command.assert_called()

    async def test_concurrent_document_processing_conflict(self):
        """동시 문서 처리 충돌 시나리오 테스트."""

        # Given: 두 개의 프로세서가 동일한 문서를 동시에 처리하려고 시도
        mock_extractor1 = AsyncMock()
        mock_extractor2 = AsyncMock()
        processor1 = DocumentProcessor(mock_extractor1, self.document_repository)
        processor2 = DocumentProcessor(mock_extractor2, self.document_repository)

        # 동시성 시뮬레이션을 위한 이벤트
        start_event = asyncio.Event()
        first_save_started = asyncio.Event()

        # 실제 동시성 충돌을 시뮬레이션하는 save 메서드
        save_call_count = 0

        async def mock_save(document_data):
            nonlocal save_call_count
            save_call_count += 1

            if save_call_count == 1:
                # 첫 번째 호출: 다른 프로세서가 시작하기를 기다림
                first_save_started.set()
                await asyncio.sleep(0.1)  # 다른 프로세서에게 기회를 줌
                return document_data
            else:
                # 두 번째 호출: 문서가 이미 존재한다고 예외 발생
                await first_save_started.wait()  # 첫 번째 저장이 시작될 때까지 대기
                raise DocumentAlreadyExistsException(document_data.id)

        async def mock_exists(document_id):
            # 실제 동시성 상황에서는 두 프로세서 모두 처음에는 문서가 없다고 판단
            await start_event.wait()  # 동시 시작을 위한 동기화
            return False

        # Mock 설정
        self.document_repository.save = AsyncMock(side_effect=mock_save)
        self.document_repository.exists = AsyncMock(side_effect=mock_exists)
        self.document_repository.update_with_knowledge = AsyncMock()

        # 지식 추출은 빠르게 완료
        mock_extractor1.extract = AsyncMock(return_value=([], []))
        mock_extractor2.extract = AsyncMock(return_value=([], []))

        # 동일한 문서 ID로 두 개의 독립적인 문서 인스턴스 생성
        document1 = Document(
            id=self.sample_document.id,
            title=self.sample_document.title,
            content=self.sample_document.content,
            doc_type=self.sample_document.doc_type,
        )

        document2 = Document(
            id=self.sample_document.id,  # 같은 ID
            title=self.sample_document.title,
            content=self.sample_document.content,
            doc_type=self.sample_document.doc_type,
        )

        # When: 두 프로세서를 동시에 실행
        async def process_document_1():
            await start_event.wait()
            return await processor1.process_document(document1)

        async def process_document_2():
            await start_event.wait()
            return await processor2.process_document(document2)

        # 동시 실행 시작
        task1 = asyncio.create_task(process_document_1())
        task2 = asyncio.create_task(process_document_2())

        # 두 태스크가 동시에 시작하도록 신호
        start_event.set()

        # Then: 하나는 성공하고 하나는 DocumentAlreadyExistsException 발생
        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # 결과 검증: 하나는 성공, 하나는 예외
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        exception_count = sum(
            1 for result in results if isinstance(result, DocumentAlreadyExistsException)
        )

        self.assertEqual(success_count, 1, "정확히 하나의 프로세서만 성공해야 함")
        self.assertEqual(
            exception_count,
            1,
            "정확히 하나의 프로세서에서 DocumentAlreadyExistsException이 발생해야 함",
        )

        # save 메서드가 두 번 호출되었는지 확인
        self.assertEqual(save_call_count, 2, "save 메서드가 두 번 호출되어야 함")

    async def test_concurrent_document_update_version_conflict(self):
        """동시 문서 업데이트 시 버전 충돌 시나리오 테스트."""

        # Given: 이미 저장된 문서가 있는 상황
        existing_document = Document(
            id=self.sample_document.id,
            title=self.sample_document.title,
            content=self.sample_document.content,
            doc_type=self.sample_document.doc_type,
        )
        existing_document.mark_as_processed()
        existing_document.version = 1

        # 두 개의 프로세서가 동일한 문서를 재처리하려고 시도
        mock_extractor1 = AsyncMock()
        mock_extractor2 = AsyncMock()
        processor1 = DocumentProcessor(mock_extractor1, self.document_repository)
        processor2 = DocumentProcessor(mock_extractor2, self.document_repository)

        # 동시성 시뮬레이션을 위한 이벤트
        start_event = asyncio.Event()
        first_update_started = asyncio.Event()

        update_call_count = 0

        async def mock_update_with_knowledge(document_data, node_ids, relationship_ids):
            nonlocal update_call_count
            update_call_count += 1

            if update_call_count == 1:
                # 첫 번째 업데이트: 잠시 대기하여 두 번째 프로세서에게 기회를 줌
                first_update_started.set()
                await asyncio.sleep(0.1)
                # 성공적으로 버전 증가
                return document_data
            # 두 번째 업데이트: 버전 충돌 발생
            await first_update_started.wait()
            raise ConcurrentModificationError(
                document_id=document_data.id,
                expected_version=1,
                actual_version=2,  # 첫 번째 프로세서가 이미 버전을 증가시킴
            )

        # Mock 설정: 문서가 이미 존재함
        self.document_repository.exists = AsyncMock(return_value=True)
        self.document_repository.update = AsyncMock()
        self.document_repository.update_with_knowledge = AsyncMock(
            side_effect=mock_update_with_knowledge
        )

        # 지식 추출은 빠르게 완료
        mock_extractor1.extract = AsyncMock(return_value=([], []))
        mock_extractor2.extract = AsyncMock(return_value=([], []))

        # 두 개의 독립적인 문서 인스턴스 (같은 ID, 같은 버전)
        document1 = Document(
            id=existing_document.id,
            title=existing_document.title,
            content=existing_document.content,
            doc_type=existing_document.doc_type,
        )
        document1.version = 1

        document2 = Document(
            id=existing_document.id,
            title=existing_document.title,
            content=existing_document.content,
            doc_type=existing_document.doc_type,
        )
        document2.version = 1

        # When: 두 프로세서를 동시에 실행
        async def process_document_1():
            await start_event.wait()
            return await processor1.process_document(document1)

        async def process_document_2():
            await start_event.wait()
            return await processor2.process_document(document2)

        task1 = asyncio.create_task(process_document_1())
        task2 = asyncio.create_task(process_document_2())

        # 동시 시작 신호
        start_event.set()

        # Then: 하나는 성공하고 하나는 ConcurrentModificationError 발생
        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # 결과 검증
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        concurrent_error_count = sum(
            1 for result in results if isinstance(result, ConcurrentModificationError)
        )

        self.assertEqual(success_count, 1, "정확히 하나의 프로세서만 성공해야 함")
        self.assertEqual(
            concurrent_error_count,
            1,
            "정확히 하나의 프로세서에서 ConcurrentModificationError가 발생해야 함",
        )

        # update_with_knowledge 메서드가 두 번 호출되었는지 확인
        self.assertEqual(update_call_count, 2, "update_with_knowledge 메서드가 두 번 호출되어야 함")

    async def test_realistic_concurrent_processing_race_condition(self):
        """실제 동시성 경쟁 조건을 시뮬레이션하는 테스트."""

        # Given: 여러 개의 프로세서가 동시에 같은 문서를 처리하려고 시도
        num_processors = 5
        processors = []
        documents = []

        for i in range(num_processors):
            mock_extractor = AsyncMock()
            mock_extractor.extract = AsyncMock(return_value=([], []))
            processor = DocumentProcessor(mock_extractor, self.document_repository)
            processors.append(processor)

            # 같은 ID를 가진 독립적인 문서 인스턴스
            document = Document(
                id=self.sample_document.id,
                title=f"동시 처리 문서 {i}",
                content=f"내용 {i}",
                doc_type=self.sample_document.doc_type,
            )
            documents.append(document)

        # 실제 경쟁 조건을 시뮬레이션하는 save 메서드
        save_call_count = 0
        successful_saves = 0

        async def mock_save(document_data):
            nonlocal save_call_count, successful_saves
            save_call_count += 1

            # 첫 번째 호출만 성공, 나머지는 DocumentAlreadyExistsException
            if successful_saves == 0:
                successful_saves += 1
                # 약간의 지연으로 다른 프로세서들이 시작할 시간을 줌
                await asyncio.sleep(0.01)
                return document_data
            raise DocumentAlreadyExistsException(document_data.id)

        # Mock 설정
        self.document_repository.save = AsyncMock(side_effect=mock_save)
        self.document_repository.exists = AsyncMock(return_value=False)
        self.document_repository.update_with_knowledge = AsyncMock()

        # When: 모든 프로세서를 동시에 실행
        tasks = [
            processor.process_document(document)
            for processor, document in zip(processors, documents)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Then: 정확히 하나만 성공하고 나머지는 DocumentAlreadyExistsException
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        exception_count = sum(
            1 for result in results if isinstance(result, DocumentAlreadyExistsException)
        )

        self.assertEqual(success_count, 1, "정확히 하나의 프로세서만 성공해야 함")
        self.assertEqual(
            exception_count,
            num_processors - 1,
            f"{num_processors - 1}개의 프로세서에서 예외가 발생해야 함",
        )
        self.assertEqual(
            save_call_count, num_processors, f"save 메서드가 {num_processors}번 호출되어야 함"
        )

        # 성공한 처리 결과 검증
        successful_results = [result for result in results if not isinstance(result, Exception)]
        self.assertEqual(len(successful_results), 1)
        self.assertIsNotNone(successful_results[0])

        # 예외 타입 검증
        exceptions = [result for result in results if isinstance(result, Exception)]
        for exception in exceptions:
            self.assertIsInstance(exception, DocumentAlreadyExistsException)
            self.assertEqual(exception.document_id, str(self.sample_document.id))

    async def test_document_reprocessing_with_persistence(self):
        """영속성을 포함한 문서 재처리 테스트."""
        # Given: 이미 처리된 문서
        self.sample_document.mark_as_processed()
        self.sample_document.add_connected_node(NodeId.generate())
        self.sample_document.add_connected_relationship(RelationshipId.generate())

        # 새로운 지식 추출 결과
        new_node_data = NodeData(
            id=str(NodeId.generate()),
            name="새로운 개체",
            node_type=DTONodeType.PERSON,
            properties={"age": 30},
        )

        self.mock_knowledge_extractor.extract = AsyncMock(return_value=([new_node_data], []))

        # Database mocking for update operations
        # 여러 호출에 대해 다른 응답 제공
        call_count = 0

        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # exists() 호출에 대한 응답
            if "SELECT 1 FROM documents WHERE id = ?" in str(args):
                return [{"1": 1}]  # 문서 존재함

            # 다른 조회에 대한 응답
            return [
                {
                    "id": str(self.sample_document.id),
                    "title": self.sample_document.title,
                    "content": self.sample_document.content,
                    "doc_type": self.sample_document.doc_type.value,
                    "status": self.sample_document.status.value,
                    "metadata": "{}",
                    "version": self.sample_document.version,  # 현재 문서와 동일한 버전
                    "created_at": self.sample_document.created_at.isoformat(),
                    "updated_at": self.sample_document.updated_at.isoformat(),
                    "processed_at": (
                        self.sample_document.processed_at.isoformat()
                        if self.sample_document.processed_at
                        else None
                    ),
                    "connected_nodes": "[]",
                    "connected_relationships": "[]",
                }
            ]

        self.mock_database.execute_query = AsyncMock(side_effect=mock_query)
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 문서 재처리
        result = await self.document_processor.reprocess_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0].name, "새로운 개체")

        # 기존 연결 정보가 초기화되고 새로운 노드만 연결됨
        self.assertEqual(len(self.sample_document.connected_nodes), 1)
        # Compare the first connected node exists
        self.assertIsNotNone(self.sample_document.connected_nodes[0])
        self.assertEqual(len(self.sample_document.connected_relationships), 0)

        # 상태가 다시 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

    async def test_batch_document_processing(self):
        """일괄 문서 처리 테스트."""
        # Given: 여러 문서
        documents = [
            Document(
                id=DocumentId.generate(),
                title=f"문서 {i}",
                content=f"내용 {i}",
                doc_type=DocumentType.TEXT,
            )
            for i in range(3)
        ]

        # 각 문서마다 다른 지식 추출 결과
        extraction_results = [
            (
                [
                    NodeData(
                        id=str(NodeId.generate()),
                        name=f"노드 {i}",
                        node_type=DTONodeType.CONCEPT,
                        properties={},
                    )
                ],
                [],
            )
            for i in range(3)
        ]

        call_count = 0

        async def mock_extract(doc):
            nonlocal call_count
            result = extraction_results[call_count]
            call_count += 1
            return result

        self.mock_knowledge_extractor.extract = AsyncMock(side_effect=mock_extract)

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])  # 문서들이 존재하지 않음
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 모든 문서 처리
        results = []
        for doc in documents:
            result = await self.document_processor.process_document(doc)
            results.append(result)

        # Then: 결과 검증
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(len(result.nodes), 1)
            self.assertEqual(result.nodes[0].name, f"노드 {i}")
            self.assertEqual(documents[i].status, DocumentStatus.PROCESSED)

        # 모든 데이터베이스 호출이 성공적으로 이루어짐
        self.assertEqual(self.mock_database.execute_command.call_count, 6)  # save + update 각 3번

    async def test_document_processing_with_empty_extraction_result(self):
        """빈 지식 추출 결과로 문서 처리 테스트."""
        # Given: 지식 추출 결과가 비어있음
        self.mock_knowledge_extractor.extract = AsyncMock(return_value=([], []))

        # Database mocking
        self.mock_database.execute_query = AsyncMock(return_value=[])
        self.mock_database.execute_command = AsyncMock(return_value=1)

        # When: 문서 처리
        result = await self.document_processor.process_document(self.sample_document)

        # Then: 결과 검증
        self.assertIsInstance(result, KnowledgeExtractionResult)
        self.assertTrue(result.is_empty())
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.relationships), 0)

        # 문서 상태는 여전히 PROCESSED로 변경됨
        self.assertEqual(self.sample_document.status, DocumentStatus.PROCESSED)

        # 연결된 요소가 없음
        self.assertEqual(len(self.sample_document.connected_nodes), 0)
        self.assertEqual(len(self.sample_document.connected_relationships), 0)


if __name__ == "__main__":
    unittest.main()
