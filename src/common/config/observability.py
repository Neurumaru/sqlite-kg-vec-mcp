"""
관찰 가능성 및 모니터링 구성 설정.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class LangfuseConfig(BaseSettings):
    """Langfuse 관찰 가능성 서비스 구성."""

    enabled: bool = Field(default=False, description="Langfuse 통합 활성화")

    host: str | None = Field(default=None, description="Langfuse 서버 호스트")

    public_key: str | None = Field(default=None, description="Langfuse 공개 키")

    secret_key: str | None = Field(default=None, description="Langfuse 비밀 키")

    project_name: str | None = Field(default=None, description="Langfuse 프로젝트 이름")

    flush_interval: float = Field(default=5.0, description="배치된 이벤트의 플러시 간격")

    debug: bool = Field(default=False, description="Langfuse 디버그 로깅 활성화")

    model_config = {"env_prefix": "LANGFUSE_", "extra": "ignore"}


class PrometheusConfig(BaseSettings):
    """Prometheus 메트릭스 구성."""

    enabled: bool = Field(default=False, description="Prometheus 메트릭스 활성화")

    port: int = Field(default=8080, description="Prometheus 메트릭스 서버 포트")

    host: str = Field(default="0.0.0.0", description="Prometheus 메트릭스 서버 호스트")

    path: str = Field(default="/metrics", description="메트릭스 엔드포인트 경로")

    namespace: str = Field(default="sqlite_kg_vec", description="메트릭스 네임스페이스")

    job_name: str = Field(default="knowledge_graph", description="메트릭스 작업 이름")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """포트 번호 유효성 검사."""
        if not 1 <= v <= 65535:
            raise ValueError("포트는 1에서 65535 사이여야 합니다")
        return v

    model_config = {"env_prefix": "PROMETHEUS_", "extra": "ignore"}


class OpenTelemetryConfig(BaseSettings):
    """OpenTelemetry 트레이싱 구성."""

    enabled: bool = Field(default=False, description="OpenTelemetry 트레이싱 활성화")

    service_name: str = Field(
        default="sqlite-kg-vec-mcp", description="트레이싱을 위한 서비스 이름"
    )

    service_version: str = Field(default="0.2.0", description="트레이싱을 위한 서비스 버전")

    endpoint: str | None = Field(default=None, description="OpenTelemetry 컬렉터 엔드포인트")

    headers: dict[str, str] = Field(
        default_factory=dict, description="트레이싱 익스포트를 위한 추가 헤더"
    )

    compression: str = Field(default="gzip", description="트레이스 익스포트를 위한 압축")

    timeout: float = Field(default=30.0, description="초 단위 익스포트 타임아웃")

    insecure: bool = Field(default=True, description="트레이싱을 위해 보안되지 않은 연결 사용")

    model_config = {"env_prefix": "OTEL_", "extra": "ignore"}


class LoggingObservabilityConfig(BaseSettings):
    """로깅 관련 관찰 가능성 구성."""

    level: str = Field(default="INFO", description="기본 로깅 레벨")

    format: str = Field(default="json", description="로그 형식 (json, text)")

    output: str = Field(default="console", description="로그 출력 (console, file)")

    file_path: str | None = Field(default=None, description="로그 파일 경로")

    include_trace: bool = Field(default=True, description="로그에 트레이스 정보 포함 여부")

    include_caller: bool = Field(default=False, description="로그에 호출자 정보 포함 여부")

    sanitize_sensitive_data: bool = Field(
        default=True, description="로그에서 민감한 데이터 삭제 여부"
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """로그 레벨 유효성 검사."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"레벨은 {valid_levels} 중 하나여야 합니다")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """로그 형식 유효성 검사."""
        valid_formats = {"json", "text"}
        if v not in valid_formats:
            raise ValueError(f"형식은 {valid_formats} 중 하나여야 합니다")
        return v

    model_config = {"env_prefix": "LOG_", "extra": "ignore"}


class ObservabilityConfig(BaseSettings):
    """
    결합된 관찰 가능성 구성 설정.

    로깅, 트레이싱, 메트릭스 및 외부 서비스에 대한 설정을 포함합니다.
    """

    # 관찰 가능성 활성화/비활성화
    enabled: bool = Field(default=True, description="관찰 가능성 기능 활성화")

    # 서비스 식별
    service_name: str = Field(
        default="sqlite-kg-vec-mcp", description="관찰 가능성을 위한 서비스 이름"
    )

    service_version: str = Field(default="0.2.0", description="관찰 가능성을 위한 서비스 버전")

    environment: str = Field(
        default="development", description="환경 (development, staging, production)"
    )

    # 컴포넌트 구성
    logging: LoggingObservabilityConfig = Field(
        default_factory=LoggingObservabilityConfig, description="로깅 구성"
    )

    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig, description="Langfuse 구성")

    prometheus: PrometheusConfig = Field(
        default_factory=PrometheusConfig, description="Prometheus 구성"
    )

    opentelemetry: OpenTelemetryConfig = Field(
        default_factory=OpenTelemetryConfig, description="OpenTelemetry 구성"
    )

    # 샘플링 및 성능
    trace_sampling_ratio: float = Field(default=1.0, description="트레이스 샘플링 비율 (0.0-1.0)")

    metrics_interval: float = Field(default=60.0, description="초 단위 메트릭스 수집 간격")

    @field_validator("trace_sampling_ratio")
    @classmethod
    def validate_trace_sampling_ratio(cls, v: float) -> float:
        """트레이스 샘플링 비율 유효성 검사."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("트레이스 샘플링 비율은 0.0에서 1.0 사이여야 합니다")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """환경 유효성 검사."""
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"환경은 {valid_envs} 중 하나여야 합니다")
        return v

    model_config = {"env_prefix": "OBSERVABILITY_", "env_file": ".env", "extra": "ignore"}
