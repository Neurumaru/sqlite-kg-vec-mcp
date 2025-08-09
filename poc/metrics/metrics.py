"""
Metrics POC - OpenTelemetry 기반 메트릭 수집 시스템의 최소 구현

이 POC는 카운터와 히스토그램을 사용한 메트릭 수집과 OTLP/Console 익스포트를 보여줍니다.
"""
import os
import time
from typing import Dict, Optional, Union

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource


class MetricsCollector:
    """
    OpenTelemetry 기반 메트릭 수집기.
    카운터와 히스토그램을 지원하며, OTLP 또는 Console로 내보냅니다.
    """

    def __init__(
        self,
        service_name: str = "metrics-poc",
        service_version: str = "0.1.0",
        endpoint: Optional[str] = None,
        export_interval_ms: int = 5000,
    ):
        """
        메트릭 수집기를 초기화합니다.
        
        Args:
            service_name: 서비스 이름
            service_version: 서비스 버전  
            endpoint: OTLP 엔드포인트 (None이면 Console 익스포터 사용)
            export_interval_ms: 메트릭 내보내기 간격 (밀리초)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.endpoint = endpoint
        self.export_interval_ms = export_interval_ms
        
        # 메트릭 저장소
        self._counters: Dict[str, metrics.Counter] = {}
        self._histograms: Dict[str, metrics.Histogram] = {}
        
        # OpenTelemetry 설정
        self._setup_otel()

    def _setup_otel(self) -> None:
        """OpenTelemetry 메트릭 프로바이더를 설정합니다."""
        # 리소스 정의
        resource = Resource(
            attributes={
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
            }
        )

        # 익스포터 선택
        if self.endpoint:
            # OTLP 익스포터 (프로덕션용)
            exporter = OTLPMetricExporter(
                endpoint=self.endpoint,
                insecure=True,  # POC용 - 프로덕션에서는 보안 설정 필요
            )
        else:
            # Console 익스포터 (개발/디버깅용)
            exporter = ConsoleMetricExporter()

        # 주기적 익스포터 설정
        metric_reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=self.export_interval_ms,
        )

        # 메터 프로바이더 설정
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter(__name__)
        
        print(f"Metrics initialized: service={self.service_name}, "
              f"endpoint={'OTLP' if self.endpoint else 'Console'}")

    def get_counter(self, name: str, description: str = "") -> metrics.Counter:
        """
        카운터 메트릭을 가져오거나 생성합니다.
        
        Args:
            name: 카운터 이름
            description: 카운터 설명
            
        Returns:
            Counter 인스턴스
        """
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(
                name=name,
                description=description or f"Counter for {name}",
            )
        return self._counters[name]

    def get_histogram(self, name: str, description: str = "") -> metrics.Histogram:
        """
        히스토그램 메트릭을 가져오거나 생성합니다.
        
        Args:
            name: 히스토그램 이름
            description: 히스토그램 설명
            
        Returns:
            Histogram 인스턴스
        """
        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                name=name,
                description=description or f"Histogram for {name}",
            )
        return self._histograms[name]

    def increment_counter(
        self, 
        name: str, 
        value: Union[int, float] = 1, 
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        카운터를 증가시킵니다.
        
        Args:
            name: 카운터 이름
            value: 증가시킬 값
            attributes: 메트릭 속성/태그
        """
        counter = self.get_counter(name)
        counter.add(value, attributes=attributes or {})

    def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        히스토그램에 값을 기록합니다.
        
        Args:
            name: 히스토그램 이름
            value: 기록할 값
            attributes: 메트릭 속성/태그
        """
        histogram = self.get_histogram(name)
        histogram.record(value, attributes=attributes or {})

    def record_timing(self, name: str, start_time: float, attributes: Optional[Dict[str, str]] = None) -> None:
        """
        시작 시간부터 현재까지의 경과 시간을 히스토그램에 기록합니다.
        
        Args:
            name: 히스토그램 이름
            start_time: 시작 시간 (time.time() 값)
            attributes: 메트릭 속성/태그
        """
        duration = time.time() - start_time
        self.record_histogram(name, duration, attributes)

    def shutdown(self) -> None:
        """메트릭 프로바이더를 종료하고 모든 데이터를 플러시합니다."""
        if hasattr(self, '_meter'):
            provider = metrics.get_meter_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()


def create_metrics_collector_from_env() -> MetricsCollector:
    """
    환경변수로부터 MetricsCollector를 생성합니다.
    
    환경변수:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP 엔드포인트 URL
        OTEL_SERVICE_NAME: 서비스 이름 (기본값: metrics-poc)
        OTEL_SERVICE_VERSION: 서비스 버전 (기본값: 0.1.0)
        METRICS_EXPORT_INTERVAL_MS: 내보내기 간격 (기본값: 5000)
    
    Returns:
        구성된 MetricsCollector 인스턴스
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME", "metrics-poc")
    service_version = os.getenv("OTEL_SERVICE_VERSION", "0.1.0")
    export_interval = int(os.getenv("METRICS_EXPORT_INTERVAL_MS", "5000"))
    
    return MetricsCollector(
        service_name=service_name,
        service_version=service_version,
        endpoint=endpoint,
        export_interval_ms=export_interval,
    )


# 사용 예제
if __name__ == "__main__":
    # Console 익스포터로 테스트
    collector = MetricsCollector()
    
    # 카운터 테스트
    collector.increment_counter("requests_total", 1, {"method": "GET", "status": "200"})
    collector.increment_counter("requests_total", 1, {"method": "POST", "status": "201"})
    collector.increment_counter("requests_total", 1, {"method": "GET", "status": "404"})
    
    # 히스토그램 테스트 (지연 시간)
    collector.record_histogram("request_duration_seconds", 0.123, {"method": "GET"})
    collector.record_histogram("request_duration_seconds", 0.456, {"method": "POST"})
    collector.record_histogram("request_duration_seconds", 0.789, {"method": "GET"})
    
    # 타이밍 테스트
    start = time.time()
    time.sleep(0.1)  # 작업 시뮬레이션
    collector.record_timing("operation_duration", start, {"operation": "test"})
    
    print("메트릭 기록 완료. 내보내기를 기다리는 중...")
    time.sleep(6)  # 내보내기 대기
    
    collector.shutdown()
    print("메트릭 수집기 종료됨.")