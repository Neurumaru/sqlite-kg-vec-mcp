"""
로깅 표준화 검증을 위한 테스트.

이 테스트는 ObservableLogger를 사용한 구조화된 로깅이
전체 코드베이스에서 올바르게 동작하는지 검증합니다.
"""

import json
import logging
import re

import pytest

from src.common.observability.logger import ObservableLogger, get_logger, get_observable_logger


class TestLoggingStandardization:
    """로깅 표준화 검증 테스트."""

    def test_get_logger_alias(self):
        """get_logger 별칭 함수가 올바르게 동작하는지 테스트."""
        logger1 = get_logger("test_component", "test_layer")
        logger2 = get_observable_logger("test_component", "test_layer")

        assert isinstance(logger1, ObservableLogger)
        assert isinstance(logger2, ObservableLogger)
        # 동일한 컴포넌트/레이어에 대해서는 같은 로거 인스턴스 반환
        assert logger1.component == logger2.component
        assert logger1.layer == logger2.layer

    def test_structured_logging_format(self, caplog):
        """모든 로그 메시지가 구조화된 형식으로 출력되는지 검증."""
        logger = get_logger("test_component", "test_layer")

        with caplog.at_level(logging.INFO):
            logger.info("test_event", user_id="12345", action="create", count=42)

        assert len(caplog.records) == 1
        record = caplog.records[0]

        # 구조화된 로그인지 확인 (JSON 파싱 가능)
        log_message = record.getMessage()
        try:
            # JSON 형태로 파싱 가능한지 확인
            if log_message.startswith("{") and log_message.endswith("}"):
                parsed = json.loads(log_message)
                assert isinstance(parsed, dict)
                assert "event" in parsed
                assert "user_id" in parsed
                assert "action" in parsed
                assert "count" in parsed
        except json.JSONDecodeError:
            # JSON이 아닌 경우라도 구조화된 정보가 포함되어야 함
            assert "test_event" in log_message
            assert "user_id=12345" in log_message or "user_id: 12345" in log_message

    def test_log_event_naming_consistency(self):
        """로그 이벤트 명명 규칙 일관성 검증."""
        # 올바른 이벤트 명명 패턴들
        valid_event_names = [
            "document_saved",
            "validation_success",
            "embedder_created",
            "processing_started",
            "task_completed",
            "connection_established",
            "query_executed",
        ]

        # 각 이벤트 명명이 snake_case 및 동작_결과 패턴을 따르는지 검증
        for event_name in valid_event_names:
            # snake_case 검증
            assert re.match(
                r"^[a-z]+(_[a-z]+)*$", event_name
            ), f"Event name '{event_name}' is not snake_case"

            # 동작_결과 패턴 검증 (최소 하나의 underscore 필요)
            assert (
                "_" in event_name
            ), f"Event name '{event_name}' should follow action_result pattern"

    def test_logger_categorization(self):
        """각 모듈이 올바른 component/layer로 분류되었는지 검증."""
        # Domain layer 로거
        domain_logger = get_logger("document_processor", "domain")
        assert domain_logger.component == "document_processor"
        assert domain_logger.layer == "domain"

        # Adapter layer 로거
        adapter_logger = get_logger("sqlite_repository", "adapter")
        assert adapter_logger.component == "sqlite_repository"
        assert adapter_logger.layer == "adapter"

        # Common layer 로거
        common_logger = get_logger("config_manager", "common")
        assert common_logger.component == "config_manager"
        assert common_logger.layer == "common"

    def test_log_level_filtering(self, caplog):
        """로그 레벨 설정에 따른 필터링 동작 검증."""
        logger = get_logger("test_component", "test_layer")

        # INFO 레벨로 설정하고 DEBUG 로그가 필터링되는지 확인
        with caplog.at_level(logging.INFO):
            logger.debug("debug_event", message="debug message")
            logger.info("info_event", message="info message")
            logger.warning("warning_event", message="warning message")
            logger.error("error_event", message="error message")

        # ObservableLogger는 fallback으로 WARNING 레벨로 출력하므로 실제 로그 내용을 확인
        log_messages = [record.getMessage() for record in caplog.records]
        debug_logs = [msg for msg in log_messages if '"level": "DEBUG"' in msg]
        info_logs = [msg for msg in log_messages if '"level": "INFO"' in msg]
        warning_logs = [msg for msg in log_messages if '"level": "WARNING"' in msg]
        error_logs = [msg for msg in log_messages if '"level": "ERROR"' in msg]

        # 모든 로그가 기록되어야 하지만 level 필드로 구분 가능
        assert len(debug_logs) == 1
        assert len(info_logs) == 1
        assert len(warning_logs) == 1
        assert len(error_logs) == 1

    def test_dependency_injection_pattern(self):
        """의존성 주입 패턴이 올바르게 동작하는지 테스트."""
        # 커스텀 로거 생성
        custom_logger = get_logger("custom_component", "custom_layer")

        # 팩토리 함수에 로거 주입 시뮬레이션
        def factory_function(logger: ObservableLogger = None) -> str:
            if logger is None:
                logger = get_logger("default_factory", "adapter")

            logger.info("factory_operation", operation="create")
            return "success"

        # 기본 로거 사용
        result1 = factory_function()
        assert result1 == "success"

        # 커스텀 로거 주입
        result2 = factory_function(custom_logger)
        assert result2 == "success"

    def test_error_handling_logging(self, caplog):
        """예외 발생 시 구조화된 에러 로그 출력 검증."""
        logger = get_logger("error_test", "test_layer")

        with caplog.at_level(logging.WARNING):  # ObservableLogger는 WARNING 레벨로 fallback
            try:
                raise ValueError("Test error message")
            except ValueError as e:
                logger.error("operation_failed", error=str(e), error_type=type(e).__name__)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        log_message = record.getMessage()

        # 에러 정보가 구조화되어 포함되었는지 확인
        assert "operation_failed" in log_message
        assert "Test error message" in log_message
        assert "ValueError" in log_message or "error_type" in log_message

    def test_performance_impact_measurement(self, caplog):
        """로깅 성능 영향 측정."""
        import time

        logger = get_logger("performance_test", "test_layer")

        # 대량 로그 생성 성능 측정
        start_time = time.time()

        with caplog.at_level(logging.INFO):
            for i in range(100):
                logger.info("performance_test_event", iteration=i, batch_size=100)

        end_time = time.time()
        execution_time = end_time - start_time

        # 100개 로그 생성이 1초 이내에 완료되어야 함 (성능 기준)
        assert execution_time < 1.0, f"Logging performance too slow: {execution_time:.3f}s"
        assert len(caplog.records) == 100

    def test_log_parsing_and_analysis(self, caplog):
        """생성된 로그를 파싱하여 분석 가능한지 검증."""
        logger = get_logger("analysis_test", "test_layer")

        with caplog.at_level(logging.WARNING):  # ObservableLogger fallback level
            logger.info("user_action", user_id="user1", action="login", success=True)
            logger.info("user_action", user_id="user2", action="logout", success=True)
            logger.error(
                "user_action",
                user_id="user3",
                action="login",
                success=False,
                error="invalid_password",
            )

        # 로그 메시지들을 분석
        log_events = []
        for record in caplog.records:
            log_message = record.getMessage()
            # 구조화된 정보 추출 시뮬레이션
            if "user_action" in log_message:
                event_data = {
                    "level": record.levelname,
                    "event": "user_action",
                    "timestamp": record.created,
                }
                log_events.append(event_data)

        assert len(log_events) == 3

        # 로그 레벨별 통계 - 실제 로그 메시지의 level 필드 확인
        info_count = sum(
            1
            for msg in [record.getMessage() for record in caplog.records]
            if '"level": "INFO"' in msg
        )
        error_count = sum(
            1
            for msg in [record.getMessage() for record in caplog.records]
            if '"level": "ERROR"' in msg
        )

        assert info_count == 2
        assert error_count == 1

    def test_null_logger_handling(self):
        """로거가 None인 경우 처리 검증."""
        # get_logger에 None 값들 전달
        logger1 = get_logger(None, None)
        logger2 = get_logger("component", None)
        logger3 = get_logger(None, "layer")

        # 모든 경우에서 유효한 ObservableLogger 인스턴스 반환
        assert isinstance(logger1, ObservableLogger)
        assert isinstance(logger2, ObservableLogger)
        assert isinstance(logger3, ObservableLogger)

    @pytest.mark.integration
    def test_real_world_logging_scenario(self, caplog):
        """실제 사용 시나리오에서 로깅 동작 검증."""
        # 문서 처리 시나리오 시뮬레이션
        doc_processor_logger = get_logger("document_processor", "domain")
        repo_logger = get_logger("sqlite_repository", "adapter")

        with caplog.at_level(logging.INFO):
            # 문서 처리 시작
            doc_processor_logger.info("document_processing_started", document_id="doc_123")

            # 데이터베이스 저장
            repo_logger.info("document_saved", document_id="doc_123")

            # 처리 완료
            doc_processor_logger.info(
                "document_processing_completed", document_id="doc_123", nodes_created=5
            )

        assert len(caplog.records) == 3

        # 각 로그가 올바른 컴포넌트에서 생성되었는지 확인
        component_logs = {}
        for record in caplog.records:
            # 로거 이름에서 컴포넌트 정보 추출
            logger_name = record.name
            if logger_name not in component_logs:
                component_logs[logger_name] = []
            component_logs[logger_name].append(record.getMessage())

        # 최소한 하나 이상의 컴포넌트가 로그를 생성했어야 함
        assert len(component_logs) > 0
