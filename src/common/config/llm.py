"""
LLM (대규모 언어 모델) 구성 설정.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class OllamaConfig(BaseSettings):
    """Ollama 서비스 구성."""

    host: str = Field(default="localhost", description="Ollama 서버 호스트")

    port: int = Field(default=11434, description="Ollama 서버 포트")

    timeout: float = Field(default=30.0, description="초 단위 요청 타임아웃")

    model: str = Field(default="llama3.2", description="기본 Ollama 모델")

    temperature: float = Field(default=0.7, description="기본 샘플링 온도")

    max_tokens: int = Field(default=2000, description="응답을 위한 최대 토큰")

    embedding_dimension: int = Field(default=768, description="임베딩 차원")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """포트 번호 유효성 검사."""
        if not 1 <= v <= 65535:
            raise ValueError("포트는 1에서 65535 사이여야 합니다")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """타임아웃 범위 유효성 검사."""
        if v <= 0:
            raise ValueError("타임아웃은 0보다 커야 합니다")
        if v > 600:
            raise ValueError("타임아웃은 10분(600초) 이하여야 합니다")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """온도 범위 유효성 검사."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("온도는 0.0에서 2.0 사이여야 합니다")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """최대 토큰 수 유효성 검사."""
        if v <= 0:
            raise ValueError("최대 토큰 수는 양수여야 합니다")
        if v > 100000:
            raise ValueError("최대 토큰 수는 100,000 이하여야 합니다")
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int) -> int:
        """임베딩 차원 유효성 검사."""
        if v <= 0:
            raise ValueError("임베딩 차원은 양수여야 합니다")
        if v > 4096:
            raise ValueError("임베딩 차원은 4096 이하여야 합니다")
        return v

    model_config = {"env_prefix": "OLLAMA_", "extra": "ignore"}


class OpenAIConfig(BaseSettings):
    """OpenAI API 구성."""

    api_key: Optional[str] = Field(default=None, description="OpenAI API 키")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """OpenAI API 키 형식 유효성 검사."""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("API 키는 비어 있지 않은 문자열이어야 합니다")
            if not v.startswith(("sk-", "sk-proj-")):
                raise ValueError("OpenAI API 키는 'sk-' 또는 'sk-proj-'로 시작해야 합니다")
        return v

    model: str = Field(default="gpt-4o-mini", description="기본 OpenAI 모델")

    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI 임베딩 모델")

    embedding_dimension: Optional[int] = Field(default=None, description="임베딩 차원 (모델별)")

    temperature: float = Field(default=0.7, description="기본 샘플링 온도")

    max_tokens: int = Field(default=2000, description="응답을 위한 최대 토큰")

    timeout: float = Field(default=30.0, description="초 단위 요청 타임아웃")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """온도 범위 유효성 검사."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("온도는 0.0에서 2.0 사이여야 합니다")
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[int]) -> Optional[int]:
        """임베딩 차원 유효성 검사."""
        if v is not None and v <= 0:
            raise ValueError("임베딩 차원은 양수여야 합니다")
        return v

    model_config = {"env_prefix": "OPENAI_", "extra": "ignore"}


class AnthropicConfig(BaseSettings):
    """Anthropic (Claude) API 구성."""

    api_key: Optional[str] = Field(default=None, description="Anthropic API 키")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Anthropic API 키 형식 유효성 검사."""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("API 키는 비어 있지 않은 문자열이어야 합니다")
            if not v.startswith("sk-ant-"):
                raise ValueError("Anthropic API 키는 'sk-ant-'로 시작해야 합니다")
        return v

    model: str = Field(default="claude-3-haiku-20240307", description="기본 Claude 모델")

    temperature: float = Field(default=0.7, description="기본 샘플링 온도")

    max_tokens: int = Field(default=2000, description="응답을 위한 최대 토큰")

    timeout: float = Field(default=30.0, description="초 단위 요청 타임아웃")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """온도 범위 유효성 검사."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("온도는 0.0에서 1.0 사이여야 합니다")
        return v

    model_config = {"env_prefix": "ANTHROPIC_", "extra": "ignore"}


class LLMConfig(BaseSettings):
    """
    결합된 LLM 구성 설정.

    지원되는 모든 LLM 제공업체에 대한 설정을 포함합니다.
    """

    # 기본 제공업체
    default_provider: str = Field(
        default="ollama", description="기본 LLM 제공업체 (ollama, openai, anthropic)"
    )

    # 제공업체 구성
    ollama: OllamaConfig = Field(default_factory=OllamaConfig, description="Ollama 구성")

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig, description="OpenAI 구성")

    anthropic: AnthropicConfig = Field(
        default_factory=AnthropicConfig, description="Anthropic 구성"
    )

    # 공통 설정
    retry_attempts: int = Field(default=3, description="실패한 요청에 대한 재시도 횟수")

    retry_delay: float = Field(default=1.0, description="초 단위 재시도 간 기본 지연")

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str) -> str:
        """기본 제공업체 유효성 검사."""
        valid_providers = {"ollama", "openai", "anthropic"}
        if v not in valid_providers:
            raise ValueError(f"제공업체는 {valid_providers} 중 하나여야 합니다")
        return v

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, v: int) -> int:
        """재시도 횟수 유효성 검사."""
        if v < 0:
            raise ValueError("재시도 횟수는 0 이상이어야 합니다")
        if v > 10:
            raise ValueError("재시도 횟수는 10회 이하여야 합니다")
        return v

    @field_validator("retry_delay")
    @classmethod
    def validate_retry_delay(cls, v: float) -> float:
        """재시도 지연 시간 유효성 검사."""
        if v < 0:
            raise ValueError("재시도 지연 시간은 0 이상이어야 합니다")
        if v > 60:
            raise ValueError("재시도 지연 시간은 60초 이하여야 합니다")
        return v

    def validate_provider_config(self) -> None:
        """선택된 제공업체가 유효한 구성을 가지고 있는지 확인합니다."""
        if self.default_provider == "openai" and not self.openai.api_key:
            raise ValueError("OpenAI 제공업체를 사용할 때는 OpenAI API 키가 필요합니다")
        if self.default_provider == "anthropic" and not self.anthropic.api_key:
            raise ValueError("Anthropic 제공업체를 사용할 때는 Anthropic API 키가 필요합니다")

    def get_active_provider_config(self) -> Union[OllamaConfig, OpenAIConfig, AnthropicConfig]:
        """활성 제공업체의 구성을 가져옵니다."""
        if self.default_provider == "openai":
            return self.openai
        if self.default_provider == "anthropic":
            return self.anthropic
        return self.ollama

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}
