"""
OpenTelemetry 공식 패턴을 따른 초기화 모듈.

공식 문서 기반: https://opentelemetry.io/docs/languages/python/getting-started/
"""

import os
from typing import Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def setup_tracing(
    service_name: str = "sqlite-kg-vec-mcp",
    service_version: str = "0.2.0",
    endpoint: str | None = None,
    insecure: bool = True,
    enable_console: bool = True,
) -> bool:
    """
    OpenTelemetry 트레이싱 설정 (공식 패턴).

    인자:
        service_name: 서비스 이름
        service_version: 서비스 버전
        endpoint: OTLP 엔드포인트
        insecure: 비보안 연결 사용
        enable_console: 콘솔 출력 활성화

    반환:
        설정 성공 여부
    """
    if not OTEL_AVAILABLE:
        print(
            "OpenTelemetry를 사용할 수 없습니다. 다음으로 설치하세요: uv add opentelemetry-api opentelemetry-sdk"
        )
        return False

    try:
        # 리소스 생성 (공식 패턴)
        resource = Resource.create(
            attributes={
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
            }
        )

        # TracerProvider 생성 및 설정
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Span Processor 추가
        if endpoint:
            # OTLP 익스포터
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=insecure,
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)

        if enable_console or not endpoint:
            # 콘솔 익스포터 (개발용)
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)

        print(f"✓ {service_name}에 대한 OpenTelemetry 트레이싱이 초기화되었습니다.")
        return True

    except Exception as exception:
        print(f"트레이싱 설정에 실패했습니다: {exception}")
        return False


def setup_metrics(
    service_name: str = "sqlite-kg-vec-mcp",
    service_version: str = "0.2.0",
    endpoint: str | None = None,
    insecure: bool = True,
    enable_console: bool = True,
) -> bool:
    """
    OpenTelemetry 메트릭 설정 (공식 패턴).

    인자:
        service_name: 서비스 이름
        service_version: 서비스 버전
        endpoint: OTLP 엔드포인트
        insecure: 비보안 연결 사용
        enable_console: 콘솔 출력 활성화

    반환:
        설정 성공 여부
    """
    if not OTEL_AVAILABLE:
        return False

    try:
        # 리소스 생성 (공식 패턴)
        resource = Resource.create(
            attributes={
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
            }
        )

        readers = []

        # OTLP 메트릭스 익스포터
        if endpoint:
            metrics_endpoint = endpoint.replace("/v1/traces", "/v1/metrics")
            otlp_exporter = OTLPMetricExporter(
                endpoint=metrics_endpoint,
                insecure=insecure,
            )
            readers.append(
                PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=30000)  # 30초
            )

        # 콘솔 메트릭스 익스포터 (개발용)
        if enable_console or not endpoint:
            console_exporter = ConsoleMetricExporter()
            readers.append(
                PeriodicExportingMetricReader(console_exporter, export_interval_millis=30000)
            )

        # MeterProvider 생성 및 설정
        meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(meter_provider)

        print(f"✓ {service_name}에 대한 OpenTelemetry 메트릭이 초기화되었습니다.")
        return True

    except Exception as exception:
        print(f"메트릭 설정에 실패했습니다: {exception}")
        return False


def setup_auto_instrumentation() -> bool:
    """
    자동 계측 설정.

    반환:
        설정 성공 여부
    """
    if not OTEL_AVAILABLE:
        return False

    success = True

    try:
        RequestsInstrumentor().instrument()
        print("✓ HTTP 요청 자동 계측 활성화됨")
    except Exception as exception:
        print(f"요청 계측에 실패했습니다: {exception}")
        success = False

    try:
        SQLite3Instrumentor().instrument()
        print("✓ SQLite3 자동 계측 활성화됨")
    except Exception as exception:
        print(f"SQLite3 계측에 실패했습니다: {exception}")
        success = False

    return success


def configure_from_env() -> dict[str, Any]:
    """
    환경변수에서 OpenTelemetry 설정 읽기 (공식 표준).

    공식 환경변수:
    - OTEL_SERVICE_NAME: 서비스 이름
    - OTEL_SERVICE_VERSION: 서비스 버전
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP 엔드포인트
    - OTEL_EXPORTER_OTLP_INSECURE: 비보안 연결 (true/false)

    반환:
        설정 딕셔너리
    """
    return {
        "service_name": os.getenv("OTEL_SERVICE_NAME", "sqlite-kg-vec-mcp"),
        "service_version": os.getenv("OTEL_SERVICE_VERSION", "0.2.0"),
        "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "insecure": os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
    }


def initialize_opentelemetry(
    service_name: str | None = None,
    service_version: str | None = None,
    endpoint: str | None = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_auto_instrumentation: bool = True,
    enable_console: bool = True,
) -> bool:
    """
    OpenTelemetry 전체 초기화 (공식 패턴 기반).

    인자:
        service_name: 서비스 이름 (None이면 환경변수 사용)
        service_version: 서비스 버전 (None이면 환경변수 사용)
        endpoint: OTLP 엔드포인트 (None이면 환경변수 사용)
        enable_tracing: 트레이싱 활성화
        enable_metrics: 메트릭 활성화
        enable_auto_instrumentation: 자동 계측 활성화
        enable_console: 콘솔 출력 활성화

    반환:
        초기화 성공 여부
    """
    if not OTEL_AVAILABLE:
        print("❌ OpenTelemetry 패키지가 설치되지 않았습니다")
        print(
            "다음으로 설치하세요: uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return False

    # 환경변수에서 설정 읽기
    env_config = configure_from_env()

    # 매개변수가 제공되지 않은 경우 환경변수 사용
    final_config = {
        "service_name": service_name or env_config["service_name"],
        "service_version": service_version or env_config["service_version"],
        "endpoint": endpoint or env_config["endpoint"],
        "insecure": env_config["insecure"],
        "enable_console": enable_console,
    }

    print(f"🚀 {final_config['service_name']}에 대한 OpenTelemetry 초기화 중...")

    success = True

    # 트레이싱 설정
    if enable_tracing:
        success &= setup_tracing(
            service_name=final_config["service_name"],
            service_version=final_config["service_version"],
            endpoint=final_config["endpoint"],
            insecure=final_config["insecure"],
        )

    # 메트릭 설정
    if enable_metrics:
        success &= setup_metrics(
            service_name=final_config["service_name"],
            service_version=final_config["service_version"],
            endpoint=final_config["endpoint"],
            insecure=final_config["insecure"],
        )

    # 자동 계측 설정
    if enable_auto_instrumentation:
        success &= setup_auto_instrumentation()

    if success:
        print("✅ OpenTelemetry 초기화가 성공적으로 완료되었습니다")
        if final_config["endpoint"]:
            print(f"📡 다음으로 익스포트 중: {final_config['endpoint']}")
        else:
            print("🖥️  콘솔 익스포트만 (개발 모드)")
    else:
        print("⚠️  OpenTelemetry 초기화 중 일부 오류가 발생했습니다")

    return success


# 편의 함수
def get_tracer(name: str = __name__):
    """트레이서 가져오기."""
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return None


def get_meter(name: str = __name__):
    """메터 가져오기."""
    if OTEL_AVAILABLE:
        return metrics.get_meter(name)
    return None
