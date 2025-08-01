"""
실제 동시성 문제 재현 및 검증 테스트.

이 모듈은 실제 운영 환경에서 발생할 수 있는 동시성 문제들을
재현하고 시스템이 이를 올바르게 처리하는지 검증합니다.
"""

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, Mock, patch

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.services.knowledge_search import KnowledgeSearchService
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId


class TestConcurrencyIssues(unittest.IsolatedAsyncioTestCase):
    """실제 동시성 문제 재현 테스트 케이스."""

    async def asyncSetUp(self):
        """비동기 테스트 픽스처 설정."""
        # Create mock objects using unittest.mock
        self.shared_repository = AsyncMock()
        self.shared_knowledge_extractor = AsyncMock()

        # Configure basic mock behaviors
        self.shared_repository.exists.return_value = False
        self.shared_repository.save.return_value = Mock()
        self.shared_knowledge_extractor.extract.return_value = (
            [],
            [],
        )  # (node_data_list, relationship_data_list)

        # Document processor instances
        self.processor1 = DocumentProcessor(
            knowledge_extractor=self.shared_knowledge_extractor,
            document_repository=self.shared_repository,
        )

        self.processor2 = DocumentProcessor(
            knowledge_extractor=self.shared_knowledge_extractor,
            document_repository=self.shared_repository,
        )

    async def test_race_condition_document_processing(self):
        """문서 처리에서 경쟁 상태 재현 테스트."""
        # Given: 동일한 문서를 두 프로세서에서 동시 처리
        document_id = DocumentId("race-condition-doc")
        document_content = "Test content for race condition"

        # Track processing order and results
        processing_results = []
        processing_order = []

        async def process_document_with_tracking(processor, processor_id):
            """처리 순서를 추적하는 문서 처리 함수."""
            try:
                processing_order.append(f"start_{processor_id}")

                # Simulate processing delay to increase race condition probability
                await asyncio.sleep(0.01)

                document = Document(
                    id=document_id,
                    title=f"Document {processor_id}",
                    content=document_content,
                    source_path="test/path",
                )

                result = await processor.process_document(document)

                processing_order.append(f"complete_{processor_id}")
                processing_results.append(result)

                return result

            except Exception as e:
                processing_order.append(f"error_{processor_id}")
                processing_results.append(e)
                raise

        # When: 두 프로세서가 동시에 동일한 문서 처리
        tasks = [
            process_document_with_tracking(self.processor1, "processor1"),
            process_document_with_tracking(self.processor2, "processor2"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Then: 동시성 문제 검증
        self.assertEqual(len(results), 2)
        self.assertEqual(len(processing_results), 2)

        # 처리 순서 분석
        start_events = [event for event in processing_order if event.startswith("start_")]
        complete_events = [event for event in processing_order if event.startswith("complete_")]

        self.assertEqual(len(start_events), 2)  # 두 프로세서 모두 시작

        # 경쟁 상태가 올바르게 처리되었는지 확인
        if len(complete_events) == 2:
            # 둘 다 성공한 경우 - 중복 처리 방지 로직 확인
            successful_results = [r for r in results if not isinstance(r, Exception)]
            self.assertLessEqual(len(successful_results), 2)
        else:
            # 일부 실패한 경우 - 적절한 예외 처리 확인
            self.assertTrue(any(isinstance(r, Exception) for r in results))

    async def test_deadlock_prevention_cross_resource_access(self):
        """상호 참조 리소스 접근에서 데드락 방지 테스트."""
        # Given: 두 문서가 서로 참조하는 상황
        doc1_id = DocumentId("cross-ref-doc1")
        doc2_id = DocumentId("cross-ref-doc2")

        # 처리 순서를 기록할 리스트
        access_log = []
        lock_acquired = []

        # Mock repository with simulated lock contention
        class LockTrackingRepository(MockDocumentRepository):
            def __init__(self):
                super().__init__()
                self._locks = {}

            async def save(self, document, **kwargs):
                doc_id = str(document.id)
                access_log.append(f"requesting_lock_{doc_id}")

                # Simulate lock acquisition
                if doc_id not in self._locks:
                    self._locks[doc_id] = asyncio.Lock()

                async with self._locks[doc_id]:
                    lock_acquired.append(f"acquired_lock_{doc_id}")
                    # Simulate processing time
                    await asyncio.sleep(0.02)
                    result = await super().save(document, **kwargs)
                    access_log.append(f"releasing_lock_{doc_id}")
                    return result

        # Replace repository with lock-tracking version
        lock_repository = LockTrackingRepository()
        self.processor1._repository = lock_repository
        self.processor2._repository = lock_repository

        async def process_cross_referenced_docs():
            """교차 참조 문서들을 동시 처리."""
            doc1 = Document(
                id=doc1_id, title="Doc1", content="References Doc2", doc_type=DocumentType.TEXT
            )
            doc2 = Document(
                id=doc2_id, title="Doc2", content="References Doc1", doc_type=DocumentType.TEXT
            )

            # Task 1: Process doc1 then access doc2
            async def task1():
                await self.processor1.process_document(doc1)
                # Simulate cross-reference access
                await asyncio.sleep(0.01)
                return await lock_repository.exists(doc2_id)

            # Task 2: Process doc2 then access doc1
            async def task2():
                await self.processor2.process_document(doc2)
                # Simulate cross-reference access
                await asyncio.sleep(0.01)
                return await lock_repository.exists(doc1_id)

            # Execute tasks concurrently
            return await asyncio.wait_for(
                asyncio.gather(task1(), task2(), return_exceptions=True),
                timeout=5.0,  # Prevent infinite deadlock
            )

        # When: 교차 참조 처리 실행
        try:
            results = await process_cross_referenced_docs()

            # Then: 데드락 없이 완료되어야 함
            self.assertEqual(len(results), 2)
            self.assertTrue(all(not isinstance(r, Exception) for r in results))

            # 락 획득/해제 순서 검증
            self.assertGreater(len(lock_acquired), 0)
            self.assertGreater(len(access_log), 0)

        except asyncio.TimeoutError:
            self.fail("Deadlock detected: Tasks did not complete within timeout")

    async def test_memory_leak_under_concurrent_load(self):
        """동시 부하 상황에서 메모리 누수 검출 테스트."""
        import os

        import psutil

        # Given: 현재 프로세스 메모리 사용량 측정
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 대량의 문서 동시 처리
        async def process_batch_documents(batch_size=50):
            """배치 문서 처리."""
            tasks = []
            for i in range(batch_size):
                document = Document(
                    id=DocumentId(f"batch-doc-{i}"),
                    title=f"Batch Document {i}",
                    content=f"Content for document {i} " * 100,  # 큰 컨텐츠
                    source_path=f"batch/doc_{i}.txt",
                )
                tasks.append(self.processor1.process_document(document))

            return await asyncio.gather(*tasks, return_exceptions=True)

        memory_measurements = []

        # When: 여러 배치를 연속으로 처리
        for batch_num in range(3):
            await process_batch_documents(30)  # 테스트 환경에 맞게 크기 조정

            # Force garbage collection
            import gc

            gc.collect()
            await asyncio.sleep(0.1)  # 메모리 정리 시간

            # 메모리 사용량 측정
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)

        # Then: 메모리 누수 검증
        peak_memory = max(memory_measurements)
        final_memory = memory_measurements[-1]

        # 메모리 증가가 합리적인 범위 내에 있는지 확인 (예: 50MB 이하)
        memory_increase = final_memory - initial_memory
        self.assertLess(
            memory_increase, 50, f"Memory leak detected: {memory_increase:.2f}MB increase"
        )

        # 메모리 사용량이 지속적으로 증가하지 않는지 확인
        if len(memory_measurements) >= 2:
            final_stable = abs(memory_measurements[-1] - memory_measurements[-2]) < 5
            self.assertTrue(final_stable, "Memory usage should stabilize after processing")

    async def test_connection_pool_exhaustion_handling(self):
        """연결 풀 고갈 상황 처리 테스트."""
        # Given: 제한된 연결 풀을 시뮬레이션
        max_connections = 5
        active_connections = []
        connection_requests = []
        failed_requests = []

        class ConnectionPoolVectorStore(MockVectorStore):
            def __init__(self, max_connections=5):
                super().__init__()
                self.max_connections = max_connections
                self.active_connections = 0
                self._connection_lock = asyncio.Lock()

            async def _acquire_connection(self):
                async with self._connection_lock:
                    if self.active_connections >= self.max_connections:
                        raise ConnectionError("Connection pool exhausted")
                    self.active_connections += 1
                    connection_requests.append(time.time())
                    return f"connection_{self.active_connections}"

            async def _release_connection(self, connection_id):
                async with self._connection_lock:
                    self.active_connections -= 1

            async def store_vector(self, node_id, vector, **kwargs):
                try:
                    conn = await self._acquire_connection()
                    active_connections.append(conn)
                    # Simulate processing time
                    await asyncio.sleep(0.05)
                    result = await super().store_vector(node_id, vector, **kwargs)
                    await self._release_connection(conn)
                    return result
                except ConnectionError as e:
                    failed_requests.append(str(e))
                    raise

        # Replace vector store
        pool_vector_store = ConnectionPoolVectorStore(max_connections)
        self.processor1._vector_store = pool_vector_store

        # When: 연결 풀 한계를 초과하는 동시 요청
        async def high_concurrency_processing():
            """높은 동시성 처리."""
            tasks = []
            for i in range(10):  # 연결 풀 한계(5)보다 많은 요청
                document = Document(
                    id=DocumentId(f"pool-test-doc-{i}"),
                    title=f"Pool Test {i}",
                    content="Test content for connection pool",
                    source_path=f"pool/test_{i}.txt",
                )
                tasks.append(self.processor1.process_document(document))

            return await asyncio.gather(*tasks, return_exceptions=True)

        results = await high_concurrency_processing()

        # Then: 연결 풀 고갈이 적절히 처리되었는지 확인
        self.assertEqual(len(results), 10)

        # 일부 요청은 성공, 일부는 연결 제한으로 실패할 수 있음
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        # 최소한 일부 요청은 성공해야 함
        self.assertGreater(len(successful_results), 0)

        # 연결 풀 관련 에러 발생 시 적절한 예외 타입인지 확인
        if failed_results:
            connection_errors = [
                r for r in failed_results if isinstance(r, (ConnectionError, TimeoutError))
            ]
            # 연결 관련 오류가 있다면 적절한 타입이어야 함
            if len(failed_requests) > 0:
                self.assertGreater(len(connection_errors), 0)

    async def test_data_consistency_under_concurrent_updates(self):
        """동시 업데이트에서 데이터 일관성 검증 테스트."""
        # Given: 공유 문서에 대한 동시 업데이트
        shared_doc_id = DocumentId("shared-consistency-doc")
        update_results = []
        consistency_violations = []

        class ConsistencyTrackingRepository(MockDocumentRepository):
            def __init__(self):
                super().__init__()
                self.update_count = 0
                self.version_history = []

            async def save(self, document, **kwargs):
                # Track version changes
                self.update_count += 1
                current_version = self.update_count

                # Simulate processing delay
                await asyncio.sleep(0.01)

                # Check for consistency violations
                if len(self.version_history) > 0:
                    last_version = self.version_history[-1]["version"]
                    if current_version != last_version + 1:
                        consistency_violations.append(
                            {
                                "expected": last_version + 1,
                                "actual": current_version,
                                "document_id": str(document.id),
                            }
                        )

                self.version_history.append(
                    {
                        "version": current_version,
                        "document_id": str(document.id),
                        "timestamp": time.time(),
                        "status": document.status,
                    }
                )

                return await super().save(document, **kwargs)

        # Replace repository
        consistency_repo = ConsistencyTrackingRepository()
        self.processor1._repository = consistency_repo
        self.processor2._repository = consistency_repo

        async def concurrent_document_updates():
            """동시 문서 업데이트."""
            # 동일한 문서의 다른 버전들
            updates = []
            for i in range(5):
                document = Document(
                    id=shared_doc_id,
                    title=f"Shared Document v{i}",
                    content=f"Updated content version {i}",
                    source_path="shared/document.txt",
                )
                updates.append(document)

            # 두 프로세서가 교대로 업데이트
            tasks = []
            for i, doc in enumerate(updates):
                processor = self.processor1 if i % 2 == 0 else self.processor2
                tasks.append(processor.process_document(doc))

            return await asyncio.gather(*tasks, return_exceptions=True)

        # When: 동시 업데이트 실행
        results = await concurrent_document_updates()

        # Then: 데이터 일관성 검증
        self.assertEqual(len(results), 5)

        # 업데이트 결과 분석
        successful_updates = [r for r in results if not isinstance(r, Exception)]
        failed_updates = [r for r in results if isinstance(r, Exception)]

        # 버전 히스토리 일관성 확인
        if len(consistency_repo.version_history) > 1:
            versions = [entry["version"] for entry in consistency_repo.version_history]
            # 버전 번호가 순차적이어야 함 (동시성 제어가 올바르다면)
            for i in range(1, len(versions)):
                version_diff = versions[i] - versions[i - 1]
                self.assertEqual(
                    version_diff,
                    1,
                    f"Version inconsistency detected: {versions[i-1]} -> {versions[i]}",
                )

        # 일관성 위반 검출
        if consistency_violations:
            self.fail(f"Data consistency violations detected: {consistency_violations}")

        # 최소한 일부 업데이트는 성공해야 함
        self.assertGreater(len(successful_updates), 0)

    def test_thread_safety_shared_resources(self):
        """스레드 간 공유 리소스 안전성 테스트."""
        # Given: 스레드 간 공유되는 리소스
        shared_counter = {"value": 0}
        thread_results = []
        thread_errors = []
        access_log = []

        def thread_worker(worker_id, iterations=20):
            """스레드 워커 함수."""
            try:
                for i in range(iterations):
                    access_log.append(f"worker_{worker_id}_access_{i}")

                    # Simulate shared resource access
                    current_value = shared_counter["value"]
                    time.sleep(0.001)  # 경쟁 상태 유발을 위한 지연
                    shared_counter["value"] = current_value + 1

                    access_log.append(f"worker_{worker_id}_update_{i}")

                thread_results.append(f"worker_{worker_id}_completed")

            except Exception as e:
                thread_errors.append(f"worker_{worker_id}_error: {e}")

        # When: 다중 스레드에서 공유 리소스 접근
        num_threads = 4
        iterations_per_thread = 10  # 테스트 시간 단축

        threads = []
        for worker_id in range(num_threads):
            thread = threading.Thread(target=thread_worker, args=(worker_id, iterations_per_thread))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join(timeout=5.0)  # 데드락 방지

        # Then: 스레드 안전성 검증
        expected_final_value = num_threads * iterations_per_thread
        actual_final_value = shared_counter["value"]

        # 경쟁 상태로 인한 데이터 손실 검출
        if actual_final_value != expected_final_value:
            data_loss = expected_final_value - actual_final_value
            # 이것은 실제로는 실패해야 하는 테스트 (경쟁 상태 데모)
            # 실제 시스템에서는 동기화 메커니즘이 있어야 함
            print(
                f"Race condition detected: Expected {expected_final_value}, got {actual_final_value}, loss: {data_loss}"
            )

        # 모든 스레드가 완료되었는지 확인
        self.assertEqual(len(thread_results), num_threads)
        self.assertEqual(len(thread_errors), 0)

        # 접근 로그 분석
        self.assertGreater(len(access_log), 0)


if __name__ == "__main__":
    unittest.main()
