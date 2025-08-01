"""
리소스 모니터링 및 정리 테스트.

이 모듈은 시스템 리소스(메모리, 연결, 파일 핸들 등)의
적절한 관리와 정리를 검증합니다.
"""

import asyncio
import os
import tempfile
import unittest
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.domain.entities.document import Document, DocumentType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId


class TestResourceManagement(unittest.IsolatedAsyncioTestCase):
    """리소스 관리 테스트 케이스."""

    async def asyncSetUp(self):
        """비동기 테스트 픽스처 설정."""
        # Create mock objects using unittest.mock
        self.repository = AsyncMock()
        self.knowledge_extractor = AsyncMock()

        # Configure basic mock behaviors
        self.repository.exists.return_value = False
        self.repository.save.return_value = Mock()
        self.knowledge_extractor.extract.return_value = (
            [],
            [],
        )  # (node_data_list, relationship_data_list)

        self.processor = DocumentProcessor(
            knowledge_extractor=self.knowledge_extractor, document_repository=self.repository
        )

        # Get current process for resource monitoring (if psutil available)
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.initial_resources = await self._get_resource_snapshot()
        else:
            self.process = None
            self.initial_resources = {}

    async def _get_resource_snapshot(self):
        """현재 리소스 사용량 스냅샷 생성."""
        if not PSUTIL_AVAILABLE or not self.process:
            return {}

        return {
            "memory_rss": self.process.memory_info().rss,
            "memory_vms": self.process.memory_info().vms,
            "open_files": len(self.process.open_files()),
            "connections": len(self.process.connections()),
            "threads": self.process.num_threads(),
        }

    async def test_memory_cleanup_after_document_processing(self):
        """문서 처리 후 메모리 정리 검증 테스트."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory monitoring")

        # Given: Initial memory state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        processed_documents = []

        # Create documents with substantial content
        large_documents = []
        for i in range(10):
            doc = Document(
                id=DocumentId(f"memory-test-doc-{i}"),
                title=f"Large Document {i}",
                content="Large content " * 1000,  # ~13KB per document
                doc_type=DocumentType.TEXT,
            )
            large_documents.append(doc)

        # When: Process documents and track memory
        memory_during_processing = []

        for doc in large_documents:
            result = await self.processor.process_document(doc)
            processed_documents.append(result)

            # Monitor memory during processing
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_during_processing.append(current_memory)

        # Force garbage collection
        import gc

        gc.collect()
        await asyncio.sleep(0.1)  # Allow cleanup time

        final_memory = self.process.memory_info().rss / 1024 / 1024

        # Then: Verify memory cleanup
        memory_increase = final_memory - initial_memory
        peak_memory = max(memory_during_processing)

        # Memory should not continuously grow
        self.assertLess(memory_increase, 20, f"Memory increase too high: {memory_increase:.2f}MB")

        # Peak memory should be reasonable
        peak_increase = peak_memory - initial_memory
        self.assertLess(peak_increase, 30, f"Peak memory usage too high: {peak_increase:.2f}MB")

        # Memory should return close to initial after cleanup
        cleanup_efficiency = (peak_memory - final_memory) / peak_memory
        self.assertGreater(cleanup_efficiency, 0.5, "Memory cleanup efficiency should be > 50%")

    async def test_file_handle_cleanup_with_temp_files(self):
        """임시 파일 처리 후 파일 핸들 정리 검증 테스트."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for file handle monitoring")

        # Given: Initial file handle count
        initial_files = len(self.process.open_files())
        temp_files_created = []

        class FileTrackingProcessor(DocumentProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.temp_files = []

            async def _create_temp_file(self, content):
                """임시 파일 생성 시뮬레이션."""
                temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
                temp_file.write(content)
                temp_file.close()
                self.temp_files.append(temp_file.name)
                return temp_file.name

            async def _cleanup_temp_files(self):
                """임시 파일 정리."""
                for temp_file in self.temp_files:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass  # File already deleted
                self.temp_files.clear()

        # Replace processor with file-tracking version
        file_processor = FileTrackingProcessor(
            repository=self.repository, llm_service=self.llm_service, vector_store=self.vector_store
        )

        # When: Process documents with temporary file creation
        try:
            for i in range(5):
                # Create temporary file
                temp_file_path = await file_processor._create_temp_file(
                    f"Temporary content for document {i}"
                )
                temp_files_created.append(temp_file_path)

                # Process document referencing temp file
                document = Document(
                    id=DocumentId(f"temp-file-doc-{i}"),
                    title=f"Document with temp file {i}",
                    content=f"Content referencing {temp_file_path}",
                    source_path=temp_file_path,
                )

                await file_processor.process_document(document)

            # Check file handles during processing
            mid_processing_files = len(self.process.open_files())

            # Clean up temporary files
            await file_processor._cleanup_temp_files()

            # Allow cleanup time
            await asyncio.sleep(0.1)

            final_files = len(self.process.open_files())

            # Then: Verify file handle cleanup
            self.assertLessEqual(
                final_files, initial_files + 2, "File handles should be cleaned up after processing"
            )

            # Verify temporary files are actually deleted
            for temp_file in temp_files_created:
                self.assertFalse(
                    os.path.exists(temp_file), f"Temporary file should be deleted: {temp_file}"
                )

        finally:
            # Cleanup any remaining temp files
            for temp_file in temp_files_created:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    async def test_async_task_cleanup_on_cancellation(self):
        """비동기 작업 취소 시 리소스 정리 검증 테스트."""
        # Given: Tasks that can be cancelled
        running_tasks = []
        cleanup_called = []

        class CleanupTrackingProcessor(DocumentProcessor):
            async def process_document(self, document):
                try:
                    # Simulate long processing
                    await asyncio.sleep(1.0)
                    return await super().process_document(document)
                except asyncio.CancelledError:
                    # Cleanup resources on cancellation
                    cleanup_called.append(f"cleanup_{document.id}")
                    raise
                finally:
                    # Always cleanup
                    cleanup_called.append(f"finally_{document.id}")

        cleanup_processor = CleanupTrackingProcessor(
            repository=self.repository, llm_service=self.llm_service, vector_store=self.vector_store
        )

        # When: Start tasks and cancel them
        documents = [
            Document(
                id=DocumentId(f"cancel-test-doc-{i}"),
                title=f"Cancellable Document {i}",
                content="Content for cancellation test",
                source_path=f"cancel/doc_{i}.txt",
            )
            for i in range(3)
        ]

        # Start tasks
        for doc in documents:
            task = asyncio.create_task(cleanup_processor.process_document(doc))
            running_tasks.append(task)

        # Let tasks start processing
        await asyncio.sleep(0.1)

        # Cancel all tasks
        for task in running_tasks:
            task.cancel()

        # Wait for cancellation to complete
        results = await asyncio.gather(*running_tasks, return_exceptions=True)

        # Then: Verify cleanup behavior
        cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
        self.assertEqual(cancelled_count, len(documents), "All tasks should be cancelled")

        # Verify cleanup was called for each cancelled task
        cleanup_entries = [entry for entry in cleanup_called if entry.startswith("cleanup_")]
        finally_entries = [entry for entry in cleanup_called if entry.startswith("finally_")]

        self.assertEqual(
            len(cleanup_entries), len(documents), "Cleanup should be called for each cancelled task"
        )
        self.assertEqual(
            len(finally_entries), len(documents), "Finally block should execute for each task"
        )

    async def test_connection_resource_management(self):
        """연결 리소스 관리 검증 테스트."""
        # Given: Connection tracking
        connections_opened = []
        connections_closed = []
        active_connections = set()

        class ConnectionTrackingVectorStore(MockVectorStore):
            def __init__(self):
                super().__init__()
                self.connection_id_counter = 0

            async def _open_connection(self):
                """연결 열기 시뮬레이션."""
                self.connection_id_counter += 1
                conn_id = f"conn_{self.connection_id_counter}"
                connections_opened.append(conn_id)
                active_connections.add(conn_id)
                return conn_id

            async def _close_connection(self, conn_id):
                """연결 닫기 시뮬레이션."""
                connections_closed.append(conn_id)
                active_connections.discard(conn_id)

            @asynccontextmanager
            async def connection_context(self):
                """연결 컨텍스트 매니저."""
                conn_id = await self._open_connection()
                try:
                    yield conn_id
                finally:
                    await self._close_connection(conn_id)

            async def store_vector(self, node_id, vector, **kwargs):
                async with self.connection_context() as conn_id:
                    # Simulate using connection
                    await asyncio.sleep(0.01)
                    return await super().store_vector(node_id, vector, **kwargs)

        # Replace vector store
        connection_store = ConnectionTrackingVectorStore()
        self.processor._vector_store = connection_store

        # When: Process multiple documents requiring connections
        documents = [
            Document(
                id=DocumentId(f"conn-test-doc-{i}"),
                title=f"Connection Test Document {i}",
                content="Content requiring vector storage",
                source_path=f"conn/test_{i}.txt",
            )
            for i in range(5)
        ]

        results = []
        for doc in documents:
            result = await self.processor.process_document(doc)
            results.append(result)

        # Allow cleanup time
        await asyncio.sleep(0.1)

        # Then: Verify connection management
        self.assertEqual(
            len(connections_opened), len(documents), "One connection should be opened per document"
        )
        self.assertEqual(
            len(connections_closed), len(documents), "All connections should be closed"
        )
        self.assertEqual(len(active_connections), 0, "No connections should remain active")

        # All processing should succeed
        successful_results = [r for r in results if r is not None]
        self.assertEqual(len(successful_results), len(documents))

    async def test_weak_reference_cleanup_validation(self):
        """약한 참조를 통한 객체 정리 검증 테스트."""
        # Given: Objects with weak references
        created_objects = []
        weak_refs = []

        def create_document_with_tracking():
            """추적 가능한 문서 객체 생성."""
            doc = Document(
                id=DocumentId("weak-ref-test-doc"),
                title="Weak Reference Test Document",
                content="Content for weak reference testing",
                doc_type=DocumentType.TEXT,
            )
            created_objects.append(doc)
            weak_refs.append(weakref.ref(doc))
            return doc

        # When: Create and process documents, then release references
        for i in range(3):
            doc = create_document_with_tracking()
            await self.processor.process_document(doc)
            # Explicitly delete reference
            del doc

        # Clear created objects list
        created_objects.clear()

        # Force garbage collection
        import gc

        gc.collect()
        await asyncio.sleep(0.1)

        # Then: Verify objects are garbage collected
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        self.assertEqual(alive_objects, 0, "All document objects should be garbage collected")

    async def test_resource_monitoring_during_batch_processing(self):
        """배치 처리 중 리소스 모니터링 테스트."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for resource monitoring")

        # Given: Resource monitoring setup
        resource_snapshots = []

        async def monitor_resources():
            """리소스 모니터링 함수."""
            while True:
                try:
                    snapshot = await self._get_resource_snapshot()
                    snapshot["timestamp"] = asyncio.get_event_loop().time()
                    resource_snapshots.append(snapshot)
                    await asyncio.sleep(0.1)  # Monitor every 100ms
                except asyncio.CancelledError:
                    break

        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_resources())

        try:
            # When: Perform batch processing
            batch_size = 20
            documents = [
                Document(
                    id=DocumentId(f"batch-monitor-doc-{i}"),
                    title=f"Batch Document {i}",
                    content=f"Batch content {i} " * 50,  # Medium-sized content
                    source_path=f"batch/monitor_{i}.txt",
                )
                for i in range(batch_size)
            ]

            # Process in smaller batches to observe resource patterns
            batch_results = []
            for i in range(0, len(documents), 5):
                batch = documents[i : i + 5]
                batch_tasks = [self.processor.process_document(doc) for doc in batch]
                batch_result = await asyncio.gather(*batch_tasks)
                batch_results.extend(batch_result)

                # Small delay between batches
                await asyncio.sleep(0.05)

        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # Then: Analyze resource patterns
        self.assertGreater(len(resource_snapshots), 5, "Should have multiple resource snapshots")

        # Check for resource leaks
        initial_snapshot = resource_snapshots[0]
        final_snapshot = resource_snapshots[-1]

        memory_growth = (
            (final_snapshot["memory_rss"] - initial_snapshot["memory_rss"]) / 1024 / 1024
        )
        file_handle_growth = final_snapshot["open_files"] - initial_snapshot["open_files"]

        self.assertLess(
            memory_growth, 50, f"Memory growth should be reasonable: {memory_growth:.2f}MB"
        )
        self.assertLessEqual(
            file_handle_growth, 2, f"File handle growth should be minimal: {file_handle_growth}"
        )

        # Verify all batch processing succeeded
        self.assertEqual(len(batch_results), batch_size)
        successful_results = [r for r in batch_results if r is not None]
        self.assertEqual(len(successful_results), batch_size)

    async def test_context_manager_resource_cleanup(self):
        """컨텍스트 매니저를 통한 리소스 정리 테스트."""
        # Given: Resource tracking
        resources_acquired = []
        resources_released = []

        class ResourceManager:
            def __init__(self, resource_id):
                self.resource_id = resource_id
                self.acquired = False

            async def __aenter__(self):
                self.acquired = True
                resources_acquired.append(self.resource_id)
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    resources_released.append(self.resource_id)
                    self.acquired = False
                return False  # Don't suppress exceptions

        # When: Use context managers for resource management
        async def process_with_resources():
            """리소스를 사용한 처리."""
            results = []

            for i in range(3):
                async with ResourceManager(f"resource_{i}") as resource:
                    # Simulate processing
                    document = Document(
                        id=DocumentId(f"ctx-mgr-doc-{i}"),
                        title=f"Context Manager Document {i}",
                        content="Content with resource management",
                        source_path=f"ctx/mgr_{i}.txt",
                    )

                    result = await self.processor.process_document(document)
                    results.append(result)

                    # Simulate potential exception
                    if i == 1:
                        try:
                            raise ValueError("Simulated processing error")
                        except ValueError:
                            pass  # Handle gracefully

            return results

        results = await process_with_resources()

        # Then: Verify resource management
        self.assertEqual(len(resources_acquired), 3, "All resources should be acquired")
        self.assertEqual(len(resources_released), 3, "All resources should be released")
        self.assertEqual(
            resources_acquired, resources_released, "Acquired and released resources should match"
        )

        # Processing should complete successfully
        self.assertEqual(len(results), 3)
        successful_results = [r for r in results if r is not None]
        self.assertEqual(len(successful_results), 3)


if __name__ == "__main__":
    unittest.main()
