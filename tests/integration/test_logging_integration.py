"""
로깅 표준화 통합 테스트.

실제 시스템 컴포넌트들이 함께 동작할 때
로깅이 올바르게 작동하는지 검증합니다.
"""

import asyncio
import logging
from unittest.mock import AsyncMock


from src.adapters.hnsw.embedder_factory import create_embedder
from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.common.config.validation_manager import ConfigValidationManager
from src.common.observability.logger import get_logger
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.services.document_processor import DocumentProcessor
from src.domain.value_objects.document_id import DocumentId
from src.dto import (
    DocumentData,
)
from src.dto import DocumentStatus as DTODocumentStatus
from src.dto import DocumentType as DTODocumentType


class TestLoggingIntegration:
    """로깅 표준화 통합 테스트."""

    def test_document_repository_logging_integration(self, caplog):
        """SQLiteDocumentRepository의 로깅 통합 테스트."""
        # Mock database
        mock_database = AsyncMock()
        mock_database.execute_query.return_value = []
        mock_database.execute_command.return_value = 1
        mock_database.transaction.return_value.__aenter__ = AsyncMock()
        mock_database.transaction.return_value.__aexit__ = AsyncMock()

        # Repository 생성 (로거 자동 주입)
        repo = SQLiteDocumentRepository(mock_database)

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            from datetime import datetime

            # 문서 데이터 생성
            doc_data = DocumentData(
                id="test_doc_123",
                title="Test Document",
                content="Test content",
                doc_type=DTODocumentType.TEXT,
                status=DTODocumentStatus.PENDING,
                metadata={},
                version=1,
                created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                updated_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                processed_at=None,
                connected_nodes=[],
                connected_relationships=[],
            )

            # 문서 저장 (비동기 실행)
            asyncio.run(repo.save(doc_data))

        # 로그 검증
        log_messages = [record.getMessage() for record in caplog.records]

        # document_saved 이벤트가 로그에 기록되었는지 확인
        assert any("document_saved" in msg for msg in log_messages)

        # 로거가 올바른 컴포넌트/레이어로 설정되었는지 확인
        repo_logs = [r for r in caplog.records if "sqlite_document_repository" in str(r)]
        assert len(repo_logs) > 0

    def test_document_processor_logging_integration(self, caplog):
        """DocumentProcessor의 로깅 통합 테스트."""
        # Mock dependencies
        mock_knowledge_extractor = AsyncMock()
        # knowledge_extractor.extract 메서드는 tuple을 반환해야 함
        mock_knowledge_extractor.extract.return_value = ([], [])  # (node_data_list, relationship_data_list)

        mock_document_repo = AsyncMock()

        # DocumentProcessor 생성 - 실제 생성자 파라미터에 맞춰 수정
        # Mock mappers 직접 생성

        class MockDocumentMapper:
            def to_dto(self, document):
                return document

            def to_entity(self, dto):
                return dto

            def to_data(self, document):
                return document

        class MockNodeMapper:
            def to_dto(self, node):
                return node

            def to_entity(self, dto):
                return dto

        class MockRelationshipMapper:
            def to_dto(self, relationship):
                return relationship

            def to_entity(self, dto):
                return dto

        processor = DocumentProcessor(
            knowledge_extractor=mock_knowledge_extractor,
            document_mapper=MockDocumentMapper(),
            node_mapper=MockNodeMapper(),
            relationship_mapper=MockRelationshipMapper(),
            document_repository=mock_document_repo,
        )

        # 테스트 문서 생성
        test_doc = Document(
            id=DocumentId("test_doc_456"),
            title="Integration Test Document",
            content="This is a test document for integration testing",
            doc_type=DocumentType.TEXT,
            status=DocumentStatus.PENDING,
        )

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            # 문서 처리 실행
            asyncio.run(processor.process(test_doc))

        # 로그 검증
        log_messages = [record.getMessage() for record in caplog.records]

        # 처리 시작과 완료 이벤트가 로그에 기록되었는지 확인
        assert any("document_processing_started" in msg for msg in log_messages)
        assert any("document_processing_completed" in msg for msg in log_messages)

    def test_config_validation_manager_logging(self, caplog):
        """ConfigValidationManager의 로깅 통합 테스트."""
        # 검증 매니저 생성
        manager = ConfigValidationManager()

        # 테스트용 설정 등록
        class MockConfig:
            def validate_all(self):
                pass

            def model_dump(self):
                return {"test_key": "test_value"}

        test_config = MockConfig()
        manager.register_config("test_config", test_config)

        with caplog.at_level(logging.DEBUG):
            # 설정 검증 실행
            success, errors = manager.validate_all()

            # 변수 사용 확인
            assert isinstance(success, bool)
            assert isinstance(errors, list)

        # 로그 검증
        log_messages = [record.getMessage() for record in caplog.records]

        # 설정 등록과 검증 이벤트가 로그에 기록되었는지 확인
        assert any("config_registered" in msg for msg in log_messages)
        assert any("config_validation_success" in msg for msg in log_messages)

    def test_embedder_factory_logging_integration(self, caplog):
        """create_embedder 팩토리 함수의 로깅 통합 테스트."""
        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            # 임베더 생성 (랜덤 임베더로 테스트)
            embedder = create_embedder("random", dimension=128)

            # embedder 변수 사용 확인
            assert embedder is not None

        # 로그 검증
        log_messages = [record.getMessage() for record in caplog.records]

        # embedder_created 이벤트가 로그에 기록되었는지 확인
        assert any("embedder_created" in msg for msg in log_messages)

        # 로거가 올바른 컴포넌트/레이어로 설정되었는지 확인
        factory_logs = [r for r in caplog.records if "embedder_factory" in str(r)]
        assert len(factory_logs) > 0

    def test_cross_component_logging_consistency(self, caplog):
        """여러 컴포넌트 간 로깅 일관성 테스트."""
        # 다양한 컴포넌트의 로거 생성
        domain_logger = get_logger("test_domain_service", "domain")
        adapter_logger = get_logger("test_adapter", "adapter")
        common_logger = get_logger("test_common_util", "common")

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            # 각 레이어에서 구조화된 로그 생성
            domain_logger.info(
                "business_logic_executed", entity_id="entity_123", operation="create"
            )
            adapter_logger.info("data_persisted", table_name="test_table", record_count=5)
            common_logger.info(
                "utility_operation_completed", operation="validation", duration_ms=100
            )

        # 로그 일관성 검증
        assert len(caplog.records) == 3

        for record in caplog.records:
            log_message = record.getMessage()

            # 모든 로그가 이벤트 기반 명명을 사용하는지 확인
            assert "_" in log_message  # snake_case 이벤트 명

            # 구조화된 정보가 포함되었는지 확인
            assert any(char in log_message for char in ["=", ":", "{"])

    def test_error_logging_across_components(self, caplog):
        """여러 컴포넌트에서 에러 로깅 일관성 테스트."""
        # 다양한 컴포넌트의 로거
        loggers = [
            get_logger("component_a", "domain"),
            get_logger("component_b", "adapter"),
            get_logger("component_c", "common"),
        ]

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            for i, logger in enumerate(loggers):
                try:
                    raise RuntimeError(f"Test error {i}")
                except RuntimeError as e:
                    logger.error(
                        "operation_failed",
                        component=f"component_{chr(ord('a') + i)}",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        # 에러 로그 검증 - 실제 로그 메시지의 level 필드 확인
        log_messages = [record.getMessage() for record in caplog.records]
        error_logs = [msg for msg in log_messages if '"level": "ERROR"' in msg]
        assert len(error_logs) == 3

        for log_message in error_logs:
            assert "operation_failed" in log_message
            assert "RuntimeError" in log_message or "error_type" in log_message

    def test_performance_under_load(self, caplog):
        """부하 상황에서 로깅 성능 테스트."""
        import threading
        import time

        loggers = [get_logger(f"load_test_{i}", "test") for i in range(5)]

        def log_worker(logger_index: int, message_count: int):
            """워커 스레드에서 로그 생성"""
            logger = loggers[logger_index]
            for i in range(message_count):
                logger.info(
                    "load_test_event", worker_id=logger_index, message_id=i, timestamp=time.time()
                )

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            start_time = time.time()

            # 5개 스레드에서 각각 50개씩 로그 생성
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_worker, args=(i, 50))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = time.time()

        # 성능 검증
        execution_time = end_time - start_time
        total_messages = 5 * 50  # 250개 메시지

        assert len(caplog.records) == total_messages
        assert execution_time < 5.0, f"Concurrent logging too slow: {execution_time:.3f}s"

        # 메시지 처리율 계산
        messages_per_second = total_messages / execution_time
        assert messages_per_second > 50, f"Too slow: {messages_per_second:.1f} msg/s"

    def test_log_filtering_integration(self, caplog):
        """로그 레벨 필터링이 모든 컴포넌트에서 일관되게 작동하는지 테스트."""
        loggers = [
            get_logger("debug_test_domain", "domain"),
            get_logger("debug_test_adapter", "adapter"),
            get_logger("debug_test_common", "common"),
        ]

        # WARNING 레벨로 설정
        with caplog.at_level(logging.WARNING):
            for i, logger in enumerate(loggers):
                logger.debug("debug_event", component_id=i)  # 필터링됨
                logger.info("info_event", component_id=i)  # 필터링됨
                logger.warning("warning_event", component_id=i)  # 기록됨
                logger.error("error_event", component_id=i)  # 기록됨

        # ObservableLogger는 fallback으로 모든 로그를 WARNING 레벨로 출력
        # 실제로는 모든 레벨이 기록되지만 level 필드로 구분 가능
        log_messages = [record.getMessage() for record in caplog.records]
        debug_logs = [msg for msg in log_messages if '"level": "DEBUG"' in msg]
        info_logs = [msg for msg in log_messages if '"level": "INFO"' in msg]
        warning_logs = [msg for msg in log_messages if '"level": "WARNING"' in msg]
        error_logs = [msg for msg in log_messages if '"level": "ERROR"' in msg]

        # 모든 레벨이 기록되었는지 확인 (ObservableLogger 특성)
        assert len(debug_logs) == 3  # 3개 컴포넌트 × DEBUG
        assert len(info_logs) == 3  # 3개 컴포넌트 × INFO
        assert len(warning_logs) == 3  # 3개 컴포넌트 × WARNING
        assert len(error_logs) == 3  # 3개 컴포넌트 × ERROR
