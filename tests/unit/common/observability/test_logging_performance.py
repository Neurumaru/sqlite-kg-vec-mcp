"""
로깅 성능 및 메모리 사용량 테스트.

구조화된 로깅 시스템의 성능 영향을 측정합니다.
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from src.common.observability.logger import get_logger


class TestLoggingPerformance:
    """로깅 성능 테스트."""

    def test_single_thread_performance(self, caplog):
        """단일 스레드에서 로깅 성능 측정."""
        logger = get_logger("perf_test", "test")
        message_count = 1000

        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()

            for i in range(message_count):
                logger.info(
                    "performance_test",
                    iteration=i,
                    batch_id="batch_001",
                    timestamp=time.time(),
                    data_size=42,
                )

            end_time = time.perf_counter()

        execution_time = end_time - start_time
        messages_per_second = message_count / execution_time

        # 성능 기준: 1000개 메시지를 1초 이내에 처리
        assert execution_time < 1.0, f"Single thread logging too slow: {execution_time:.3f}s"
        assert messages_per_second > 1000, f"Too slow: {messages_per_second:.1f} msg/s"
        assert len(caplog.records) == message_count

    def test_multi_thread_performance(self, caplog):
        """멀티 스레드에서 로깅 성능 측정."""

        def worker(thread_id: int, message_count: int):
            """워커 스레드 함수."""
            logger = get_logger(f"perf_worker_{thread_id}", "test")
            for i in range(message_count):
                logger.info(
                    "multi_thread_test", thread_id=thread_id, message_id=i, worker_data="test_data"
                )

        thread_count = 4
        messages_per_thread = 250
        total_messages = thread_count * messages_per_thread

        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                for thread_id in range(thread_count):
                    future = executor.submit(worker, thread_id, messages_per_thread)
                    futures.append(future)

                # 모든 스레드 완료 대기
                for future in futures:
                    future.result()

            end_time = time.perf_counter()

        execution_time = end_time - start_time
        messages_per_second = total_messages / execution_time

        # 멀티 스레드 성능 기준
        assert execution_time < 2.0, f"Multi-thread logging too slow: {execution_time:.3f}s"
        assert messages_per_second > 500, f"Too slow: {messages_per_second:.1f} msg/s"
        assert len(caplog.records) == total_messages

    def test_memory_usage_measurement(self):
        """로깅 시 메모리 사용량 측정."""
        import tracemalloc

        logger = get_logger("memory_test", "test")

        # 메모리 추적 시작
        tracemalloc.start()

        # 가비지 컬렉션으로 초기 상태 정리
        gc.collect()

        # 초기 메모리 스냅샷
        snapshot1 = tracemalloc.take_snapshot()

        # 대량 로그 생성
        for i in range(5000):
            logger.info(
                "memory_test_event",
                iteration=i,
                large_data="x" * 100,  # 100자 문자열
                metadata={"key": f"value_{i}", "index": i},
            )

        # 최종 메모리 스냅샷
        snapshot2 = tracemalloc.take_snapshot()

        # 메모리 사용량 차이 계산
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_memory_diff = sum(stat.size_diff for stat in top_stats)

        tracemalloc.stop()

        # 메모리 사용량 기준: 5000개 로그로 10MB 미만 증가
        memory_mb = total_memory_diff / (1024 * 1024)
        assert memory_mb < 10.0, f"Memory usage too high: {memory_mb:.2f}MB"

    def test_logging_overhead_comparison(self, caplog):
        """로깅 오버헤드와 비-로깅 작업 비교."""
        logger = get_logger("overhead_test", "test")
        iterations = 10000

        # 로깅 없는 작업 시간 측정
        start_time = time.perf_counter()
        for i in range(iterations):
            # 단순 연산 수행
            result = i * 2 + 1
            _ = {"iteration": i, "result": result}  # 사용하지 않는 변수
        end_time = time.perf_counter()
        no_logging_time = end_time - start_time

        # 로깅 포함 작업 시간 측정
        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()
            for i in range(iterations):
                result = i * 2 + 1
                _ = {"iteration": i, "result": result}  # 사용하지 않는 변수
                logger.info("overhead_test_operation", iteration=i, result=result)
            end_time = time.perf_counter()
            with_logging_time = end_time - start_time

        # 오버헤드 계산
        overhead_ratio = with_logging_time / no_logging_time

        # 오버헤드가 300배를 넘지 않아야 함 (구조화된 로깅의 특성상)
        assert overhead_ratio < 300.0, f"Logging overhead too high: {overhead_ratio:.2f}x"
        assert len(caplog.records) == iterations

    def test_large_message_performance(self, caplog):
        """큰 크기 로그 메시지 성능 테스트."""
        logger = get_logger("large_msg_test", "test")

        # 다양한 크기의 메시지 테스트
        message_sizes = [100, 1000, 10000, 50000]  # 문자 수

        with caplog.at_level(logging.INFO):
            for size in message_sizes:
                large_content = "x" * size

                start_time = time.perf_counter()
                logger.info(
                    "large_message_test",
                    content_size=size,
                    large_content=large_content,
                    metadata={"size": size},
                )
                end_time = time.perf_counter()

                # 개별 메시지 처리 시간이 0.1초를 넘지 않아야 함
                processing_time = end_time - start_time
                assert (
                    processing_time < 0.1
                ), f"Large message ({size} chars) too slow: {processing_time:.3f}s"

        assert len(caplog.records) == len(message_sizes)

    def test_concurrent_logger_creation_performance(self):
        """동시 로거 생성 성능 테스트."""

        def create_loggers_worker(worker_id: int, logger_count: int) -> List[str]:
            """워커에서 로거들을 생성하고 이름 반환."""
            logger_names = []
            for i in range(logger_count):
                component_name = f"worker_{worker_id}_component_{i}"
                _ = get_logger(component_name, "test")  # 로거 생성만 테스트
                logger_names.append(f"{component_name}_test")
            return logger_names

        thread_count = 8
        loggers_per_thread = 50

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for worker_id in range(thread_count):
                future = executor.submit(create_loggers_worker, worker_id, loggers_per_thread)
                futures.append(future)

            # 모든 결과 수집
            all_logger_names = []
            for future in futures:
                logger_names = future.result()
                all_logger_names.extend(logger_names)

        end_time = time.perf_counter()

        execution_time = end_time - start_time
        total_loggers = thread_count * loggers_per_thread
        loggers_per_second = total_loggers / execution_time

        # 로거 생성 성능 기준
        assert execution_time < 5.0, f"Logger creation too slow: {execution_time:.3f}s"
        assert loggers_per_second > 50, f"Logger creation rate too low: {loggers_per_second:.1f}/s"
        assert len(all_logger_names) == total_loggers

    def test_structured_vs_simple_logging_performance(self, caplog):
        """구조화된 로깅 vs 단순 로깅 성능 비교."""
        logger = get_logger("comparison_test", "test")
        iterations = 5000

        # 구조화된 로깅 성능 측정
        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()
            for i in range(iterations):
                logger.info(
                    "structured_log_test",
                    iteration=i,
                    user_id=f"user_{i % 100}",
                    action="test_action",
                    success=True,
                    duration_ms=i * 0.1,
                )
            end_time = time.perf_counter()
            structured_time = end_time - start_time

        caplog.clear()

        # 단순 로깅 성능 측정 (문자열 포맷)
        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()
            for i in range(iterations):
                logger.info(
                    f"Simple log test iteration {i} user_user_{i % 100} action_test_action success_True duration_{i * 0.1}ms"
                )
            end_time = time.perf_counter()
            simple_time = end_time - start_time

        # 성능 차이 분석
        performance_ratio = structured_time / simple_time

        # 구조화된 로깅이 단순 로깅보다 3배 이상 느리지 않아야 함
        assert (
            performance_ratio < 3.0
        ), f"Structured logging too slow vs simple: {performance_ratio:.2f}x"

        # 두 방식 모두 합리적인 성능을 보여야 함
        assert structured_time < 2.0, f"Structured logging too slow: {structured_time:.3f}s"
        assert simple_time < 2.0, f"Simple logging too slow: {simple_time:.3f}s"

    @pytest.mark.slow
    def test_sustained_logging_performance(self, caplog):
        """지속적인 로깅 성능 테스트 (장시간)."""
        logger = get_logger("sustained_test", "test")

        # 10초간 지속적으로 로깅
        test_duration = 10.0  # 초
        message_count = 0

        with caplog.at_level(logging.INFO):
            start_time = time.perf_counter()
            current_time = start_time

            while (current_time - start_time) < test_duration:
                logger.info(
                    "sustained_logging_test",
                    message_id=message_count,
                    elapsed_time=current_time - start_time,
                    timestamp=time.time(),
                )
                message_count += 1
                current_time = time.perf_counter()

            end_time = current_time

        actual_duration = end_time - start_time
        messages_per_second = message_count / actual_duration

        # 지속 성능 기준: 최소 100 msg/s 유지
        assert (
            messages_per_second > 100
        ), f"Sustained performance too low: {messages_per_second:.1f} msg/s"
        assert len(caplog.records) == message_count

        # 메모리 누수 체크를 위한 가비지 컬렉션
        gc.collect()

        # 테스트 완료 후 로그 레코드 수가 예상과 일치하는지 확인
        assert len(caplog.records) == message_count
