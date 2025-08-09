"""
Metrics POC 테스트 모듈

이 테스트는 MetricsCollector의 기본 기능을 검증합니다:
- 카운터 생성 및 증가
- 히스토그램 생성 및 값 기록
- 환경변수 기반 설정
- 메트릭 내보내기 동작
"""
import os
import time
import unittest
from unittest.mock import patch

from metrics import MetricsCollector, create_metrics_collector_from_env


class TestMetricsCollector(unittest.TestCase):
    """MetricsCollector 클래스의 단위 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.collector = MetricsCollector(
            service_name="test-service",
            service_version="0.1.0",
            endpoint=None,  # Console 익스포터 사용
            export_interval_ms=1000,  # 빠른 테스트를 위해 1초
        )

    def tearDown(self):
        """각 테스트 후에 실행되는 정리"""
        if hasattr(self, 'collector'):
            self.collector.shutdown()

    def test_initialization(self):
        """MetricsCollector가 올바르게 초기화되는지 테스트"""
        self.assertEqual(self.collector.service_name, "test-service")
        self.assertEqual(self.collector.service_version, "0.1.0")
        self.assertIsNone(self.collector.endpoint)
        self.assertEqual(self.collector.export_interval_ms, 1000)
        self.assertTrue(hasattr(self.collector, '_meter'))
        
    def test_counter_creation_and_increment(self):
        """카운터 생성 및 증가 테스트"""
        # 카운터 생성
        counter_name = "test_counter"
        counter = self.collector.get_counter(counter_name, "Test counter description")
        
        # 같은 이름으로 다시 요청했을 때 같은 인스턴스 반환 확인
        same_counter = self.collector.get_counter(counter_name)
        self.assertIs(counter, same_counter)
        
        # 카운터 증가 (에러 없이 실행되어야 함)
        self.collector.increment_counter(counter_name, 1)
        self.collector.increment_counter(counter_name, 5, {"label": "test"})

    def test_histogram_creation_and_record(self):
        """히스토그램 생성 및 값 기록 테스트"""
        # 히스토그램 생성
        histogram_name = "test_histogram"
        histogram = self.collector.get_histogram(histogram_name, "Test histogram description")
        
        # 같은 이름으로 다시 요청했을 때 같은 인스턴스 반환 확인
        same_histogram = self.collector.get_histogram(histogram_name)
        self.assertIs(histogram, same_histogram)
        
        # 히스토그램에 값 기록 (에러 없이 실행되어야 함)
        self.collector.record_histogram(histogram_name, 0.123)
        self.collector.record_histogram(histogram_name, 0.456, {"label": "test"})

    def test_timing_record(self):
        """타이밍 기록 테스트"""
        start_time = time.time()
        time.sleep(0.01)  # 10ms 대기
        
        # 타이밍 기록 (에러 없이 실행되어야 함)
        self.collector.record_timing("test_timing", start_time, {"operation": "test"})

    def test_multiple_metrics(self):
        """여러 메트릭이 동시에 정상 동작하는지 테스트"""
        # 카운터들
        self.collector.increment_counter("http_requests", 1, {"method": "GET"})
        self.collector.increment_counter("http_requests", 1, {"method": "POST"})
        self.collector.increment_counter("errors_total", 1, {"type": "connection"})
        
        # 히스토그램들
        self.collector.record_histogram("request_duration", 0.123, {"endpoint": "/api/users"})
        self.collector.record_histogram("request_duration", 0.456, {"endpoint": "/api/posts"})
        self.collector.record_histogram("db_query_duration", 0.789, {"table": "users"})


class TestEnvironmentConfiguration(unittest.TestCase):
    """환경변수 기반 설정 테스트"""

    def test_create_from_env_defaults(self):
        """기본 환경변수 설정으로 생성 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            collector = create_metrics_collector_from_env()
            
            self.assertEqual(collector.service_name, "metrics-poc")
            self.assertEqual(collector.service_version, "0.1.0")
            self.assertIsNone(collector.endpoint)
            self.assertEqual(collector.export_interval_ms, 5000)
            
            collector.shutdown()

    def test_create_from_env_custom(self):
        """커스텀 환경변수 설정으로 생성 테스트"""
        env_vars = {
            'OTEL_SERVICE_NAME': 'custom-service',
            'OTEL_SERVICE_VERSION': '1.2.3',
            'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317',
            'METRICS_EXPORT_INTERVAL_MS': '2000',
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            collector = create_metrics_collector_from_env()
            
            self.assertEqual(collector.service_name, "custom-service")
            self.assertEqual(collector.service_version, "1.2.3")
            self.assertEqual(collector.endpoint, "http://localhost:4317")
            self.assertEqual(collector.export_interval_ms, 2000)
            
            collector.shutdown()


class TestMetricsIntegration(unittest.TestCase):
    """통합 테스트"""

    def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        collector = MetricsCollector(export_interval_ms=500)  # 빠른 내보내기
        
        try:
            # 실제 사용 시나리오 시뮬레이션
            for i in range(3):
                # HTTP 요청 시뮬레이션
                start = time.time()
                
                # 요청 카운터 증가
                method = "GET" if i % 2 == 0 else "POST"
                status = "200" if i < 2 else "500"
                collector.increment_counter("http_requests_total", 1, {
                    "method": method,
                    "status": status
                })
                
                # 작업 시간 시뮬레이션
                time.sleep(0.01)
                
                # 응답 시간 기록
                collector.record_timing("http_request_duration_seconds", start, {
                    "method": method,
                    "status": status
                })
                
                # 데이터베이스 쿼리 시뮬레이션
                collector.record_histogram("db_query_duration_seconds", 0.005 * (i + 1), {
                    "table": "users",
                    "operation": "select"
                })
            
            # 내보내기 대기
            time.sleep(1)
            
        finally:
            collector.shutdown()


def run_demo():
    """실제 메트릭 내보내기를 보여주는 데모"""
    print("=== Metrics POC 데모 시작 ===")
    
    collector = MetricsCollector(
        service_name="metrics-demo",
        service_version="0.1.0",
        export_interval_ms=2000,
    )
    
    try:
        print("메트릭 생성 중...")
        
        # 카운터 메트릭
        for i in range(5):
            collector.increment_counter("demo_requests", 1, {
                "method": "GET" if i % 2 == 0 else "POST",
                "status": "200" if i < 4 else "500"
            })
        
        # 히스토그램 메트릭
        latencies = [0.050, 0.120, 0.080, 0.200, 0.035]
        for i, latency in enumerate(latencies):
            collector.record_histogram("demo_latency_seconds", latency, {
                "service": "api",
                "endpoint": f"/endpoint{i % 2}"
            })
        
        print("메트릭 내보내기 대기 중... (3초)")
        time.sleep(3)
        
    finally:
        print("데모 종료 - 메트릭 시스템 종료 중...")
        collector.shutdown()
        print("=== 데모 완료 ===")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # 데모 실행
        run_demo()
    else:
        # 단위 테스트 실행
        unittest.main()