"""
로깅 표준화 회귀 테스트.

로깅 변경이 기존 기능에 영향을 주지 않는지 검증합니다.
"""

import asyncio
import logging


from src.adapters.hnsw.embedder_factory import create_embedder
from src.common.config.validation_manager import ConfigValidationManager
from src.common.observability.logger import get_logger
from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.vector import Vector


class TestLoggingRegression:
    """로깅 표준화 회귀 테스트."""

    def test_document_entity_functionality_preserved(self, caplog):
        """Document 엔티티 기본 기능이 로깅 변경 후에도 유지되는지 확인."""
        # Document 생성 및 기본 동작 확인
        doc = Document(
            id=DocumentId("regression_test_doc"),
            title="Regression Test Document",
            content="Content for regression testing",
            doc_type=DocumentType.TEXT,
            status=DocumentStatus.PENDING,
        )

        # 기본 속성 접근
        assert str(doc.id) == "regression_test_doc"
        assert doc.title == "Regression Test Document"
        assert doc.status == DocumentStatus.PENDING

        # 상태 변경 동작
        doc.mark_as_processing()
        assert doc.status == DocumentStatus.PROCESSING

        doc.mark_as_processed()
        assert doc.status == DocumentStatus.PROCESSED

        # 버전 관리 동작
        initial_version = doc.version
        doc.increment_version()
        assert doc.version == initial_version + 1

        # 로깅으로 인한 부작용이 없는지 확인
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_embedder_factory_backward_compatibility(self, caplog):
        """임베더 팩토리의 기존 API 호환성 확인."""
        with caplog.at_level(logging.INFO):
            # 기존 방식으로 임베더 생성
            embedder1 = create_embedder("random", dimension=128)

            # 새로운 로거 주입 방식
            custom_logger = get_logger("test_embedder", "adapter")
            embedder2 = create_embedder("random", logger=custom_logger, dimension=256)

        # 두 임베더 모두 정상 동작해야 함
        assert embedder1 is not None
        assert embedder2 is not None

        # 기본 임베딩 기능 확인
        test_text = "Test embedding functionality"
        vector1 = embedder1.embed(test_text)
        vector2 = embedder2.embed(test_text)

        # RandomVectorEmbedder는 list를 반환하므로 Vector 객체로 변환 필요
        if isinstance(vector1, list):
            vector1 = Vector(values=vector1)
        if isinstance(vector2, list):
            vector2 = Vector(values=vector2)

        assert isinstance(vector1, Vector)
        assert isinstance(vector2, Vector)
        assert vector1.dimension == 128
        assert vector2.dimension == 256

    def test_config_validation_manager_existing_functionality(self, caplog):
        """ConfigValidationManager의 기존 기능이 유지되는지 확인."""
        # 기존 방식으로 매니저 생성
        manager = ConfigValidationManager()

        # 테스트용 설정 클래스
        class TestConfig:
            def __init__(self):
                self.test_value = "test"

            def validate_all(self):
                if not hasattr(self, "test_value"):
                    raise ValueError("test_value is required")

            def model_dump(self):
                return {"test_value": self.test_value}

        test_config = TestConfig()

        with caplog.at_level(logging.DEBUG):
            # 설정 등록 및 검증
            manager.register_config("test_config", test_config)
            success, errors = manager.validate_all()

        # 기존 기능이 정상 동작해야 함
        assert success is True
        assert len(errors) == 0

        # 설정 요약 기능 확인
        summary = manager.get_config_summary()
        assert "test_config" in summary
        assert summary["test_config"]["test_value"] == "test"

    def test_vector_operations_unchanged(self, caplog):
        """Vector 클래스의 기본 연산이 변경되지 않았는지 확인."""
        # Vector 생성 및 기본 연산
        vector1 = Vector(values=[1.0, 2.0, 3.0])
        vector2 = Vector(values=[2.0, 4.0, 6.0])

        # 기본 속성
        assert vector1.dimension == 3
        assert len(vector1.values) == 3

        # 정규화 기능
        normalized = vector1.normalize()
        assert isinstance(normalized, Vector)
        assert normalized.dimension == 3

        # 유사도 계산 기능
        similarity = vector1.cosine_similarity(vector2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_logging_does_not_affect_async_operations(self, caplog):
        """비동기 작업이 로깅 변경으로 인해 영향받지 않는지 확인."""

        async def async_test_operation():
            """테스트용 비동기 작업."""
            logger = get_logger("async_test", "test")

            # 비동기 작업 중 로깅
            logger.info("async_operation_started", operation_id="test_001")

            # 실제 비동기 작업 시뮬레이션
            await asyncio.sleep(0.01)

            logger.info("async_operation_completed", operation_id="test_001", duration_ms=10)
            return "success"

        with caplog.at_level(logging.INFO):
            # 비동기 작업 실행
            result = asyncio.run(async_test_operation())

        # 비동기 작업이 정상 완료되어야 함
        assert result == "success"

        # 로그가 정상 기록되어야 함
        log_messages = [record.getMessage() for record in caplog.records]
        assert any("async_operation_started" in msg for msg in log_messages)
        assert any("async_operation_completed" in msg for msg in log_messages)

    def test_exception_handling_unchanged(self, caplog):
        """예외 처리 로직이 로깅 변경으로 영향받지 않는지 확인."""
        logger = get_logger("exception_test", "test")

        # 커스텀 예외 클래스
        class CustomTestException(Exception):
            def __init__(self, message: str, error_code: int = 0):
                super().__init__(message)
                self.error_code = error_code

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            # 예외 발생 및 처리
            try:
                raise CustomTestException("Test exception for regression testing", 500)
            except CustomTestException as e:
                logger.error(
                    "custom_exception_caught",
                    error=str(e),
                    error_code=e.error_code,
                    exception_type=type(e).__name__,
                )

        # 예외가 정상 처리되고 로그에 기록되어야 함
        log_messages = [record.getMessage() for record in caplog.records]
        error_logs = [msg for msg in log_messages if '"level": "ERROR"' in msg]
        assert len(error_logs) == 1

        log_message = error_logs[0]
        assert "custom_exception_caught" in log_message
        assert "Test exception for regression testing" in log_message

    def test_memory_management_unchanged(self, caplog):
        """메모리 관리가 로깅 변경으로 영향받지 않는지 확인."""
        import gc
        import weakref

        # 객체 생성 및 참조 관리 테스트
        logger = get_logger("memory_test", "test")

        class TestObject:
            def __init__(self, data: str):
                self.data = data
                logger.debug("test_object_created", data_length=len(data))

            def __del__(self):
                logger.debug(
                    "test_object_destroyed",
                    data_length=len(self.data) if hasattr(self, "data") else 0,
                )

        with caplog.at_level(logging.DEBUG):
            # 객체 생성
            obj = TestObject("test data for memory management")
            weak_ref = weakref.ref(obj)

            # 객체가 존재하는지 확인
            assert weak_ref() is not None
            assert obj.data == "test data for memory management"

            # 객체 참조 해제
            del obj
            gc.collect()

            # 가비지 컬렉션 후 객체가 정리되었는지 확인
            assert weak_ref() is None

        # 생성/소멸 로그 확인
        log_messages = [record.getMessage() for record in caplog.records]
        assert any("test_object_created" in msg for msg in log_messages)

    def test_threading_safety_preserved(self, caplog):
        """스레딩 안전성이 로깅 변경으로 영향받지 않는지 확인."""
        import threading
        import time

        results = []
        errors = []

        def thread_worker(worker_id: int):
            """스레드 워커 함수."""
            try:
                logger = get_logger(f"thread_worker_{worker_id}", "test")

                for i in range(10):
                    logger.info("thread_operation", worker_id=worker_id, iteration=i)

                    # 실제 작업 시뮬레이션
                    time.sleep(0.001)

                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {str(e)}")

        with caplog.at_level(logging.INFO):
            # 여러 스레드 동시 실행
            threads = []
            for i in range(5):
                thread = threading.Thread(target=thread_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # 모든 스레드 완료 대기
            for thread in threads:
                thread.join()

        # 모든 스레드가 정상 완료되어야 함
        assert len(errors) == 0
        assert len(results) == 5

        # 각 워커의 모든 로그가 기록되어야 함
        assert len(caplog.records) == 5 * 10  # 5개 워커 × 10개 반복

    def test_serialization_compatibility(self, caplog):
        """직렬화 호환성이 유지되는지 확인."""
        import json

        # Document 객체 생성
        doc = Document(
            id=DocumentId("serialization_test"),
            title="Serialization Test",
            content="Test content",
            doc_type=DocumentType.TEXT,
            status=DocumentStatus.PENDING,
        )

        logger = get_logger("serialization_test", "test")

        with caplog.at_level(logging.INFO):
            # 직렬화 테스트
            logger.info("serialization_test_started", document_id=str(doc.id))

            # 기본 속성 직렬화 (JSON)
            doc_dict = {
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content,
                "doc_type": doc.doc_type.value,
                "status": doc.status.value,
                "version": doc.version,
            }

            json_str = json.dumps(doc_dict)
            restored_dict = json.loads(json_str)

            assert restored_dict["id"] == "serialization_test"
            assert restored_dict["title"] == "Serialization Test"

            logger.info("serialization_test_completed", format="json", size=len(json_str))

        # 로그 확인
        log_messages = [record.getMessage() for record in caplog.records]
        assert any("serialization_test_started" in msg for msg in log_messages)
        assert any("serialization_test_completed" in msg for msg in log_messages)

    def test_existing_test_compatibility(self, caplog):
        """기존 테스트들이 로깅 변경으로 실패하지 않는지 확인."""
        # 기존 스타일의 테스트 코드 패턴들이 여전히 동작하는지 확인

        # 1. 기본 객체 생성 및 속성 확인
        doc = Document(
            id=DocumentId("compatibility_test"),
            title="Compatibility Test",
            content="Test content",
            doc_type=DocumentType.TEXT,
            status=DocumentStatus.PENDING,
        )

        assert doc.id.value == "compatibility_test"
        assert doc.status == DocumentStatus.PENDING

        # 2. Vector 생성 및 연산
        vector = Vector(values=[1.0, 0.0, 0.0])
        assert vector.magnitude() > 0

        # 3. 임베더 생성 및 사용
        embedder = create_embedder("random", dimension=10)
        test_vector_raw = embedder.embed("test text")
        # RandomVectorEmbedder는 list를 반환하므로 Vector 객체로 변환
        if isinstance(test_vector_raw, list):
            test_vector = Vector(values=test_vector_raw)
        else:
            test_vector = test_vector_raw
        assert test_vector.dimension == 10

        # 4. ConfigValidationManager 기본 사용법
        manager = ConfigValidationManager()
        assert manager is not None

        # 로깅이 있어도 기존 기능들이 모두 정상 동작해야 함
        # 이는 로깅 추가가 기존 코드의 동작을 변경하지 않았음을 의미
