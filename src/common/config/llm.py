"""
LLM (Large Language Model) configuration settings.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class OllamaConfig(BaseSettings):
    """Ollama service configuration."""

    host: str = Field(default="localhost", description="Ollama server host")

    port: int = Field(default=11434, description="Ollama server port")

    timeout: float = Field(default=30.0, description="Request timeout in seconds")

    model: str = Field(default="llama3.2", description="Default Ollama model")

    temperature: float = Field(default=0.7, description="Default sampling temperature")

    max_tokens: int = Field(default=2000, description="Maximum tokens for responses")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    model_config = {"env_prefix": "OLLAMA_", "extra": "ignore"}


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str | None = Field(default=None, description="OpenAI API key")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate OpenAI API key format."""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("API key must be a non-empty string")
            if not v.startswith(("sk-", "sk-proj-")):
                raise ValueError("OpenAI API key must start with 'sk-' or 'sk-proj-'")
        return v

    model: str = Field(default="gpt-4o-mini", description="Default OpenAI model")

    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    embedding_dimension: int | None = Field(
        default=None, description="Embedding dimension (model-specific)"
    )

    temperature: float = Field(default=0.7, description="Default sampling temperature")

    max_tokens: int = Field(default=2000, description="Maximum tokens for responses")

    timeout: float = Field(default=30.0, description="Request timeout in seconds")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int | None) -> int | None:
        """Validate embedding dimension."""
        if v is not None and v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    model_config = {"env_prefix": "OPENAI_", "extra": "ignore"}


class AnthropicConfig(BaseSettings):
    """Anthropic (Claude) API configuration."""

    api_key: str | None = Field(default=None, description="Anthropic API key")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate Anthropic API key format."""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("API key must be a non-empty string")
            if not v.startswith("sk-ant-"):
                raise ValueError("Anthropic API key must start with 'sk-ant-'")
        return v

    model: str = Field(default="claude-3-haiku-20240307", description="Default Claude model")

    temperature: float = Field(default=0.7, description="Default sampling temperature")

    max_tokens: int = Field(default=2000, description="Maximum tokens for responses")

    timeout: float = Field(default=30.0, description="Request timeout in seconds")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v

    model_config = {"env_prefix": "ANTHROPIC_", "extra": "ignore"}


class LLMConfig(BaseSettings):
    """
    Combined LLM configuration settings.

    Includes settings for all supported LLM providers.
    """

    # Default provider
    default_provider: str = Field(
        default="ollama", description="Default LLM provider (ollama, openai, anthropic)"
    )

    # Provider configs
    ollama: OllamaConfig = Field(default_factory=OllamaConfig, description="Ollama configuration")

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig, description="OpenAI configuration")

    anthropic: AnthropicConfig = Field(
        default_factory=AnthropicConfig, description="Anthropic configuration"
    )

    # Common settings
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed requests"
    )

    retry_delay: float = Field(default=1.0, description="Base delay between retries in seconds")

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str) -> str:
        """Validate default provider."""
        valid_providers = {"ollama", "openai", "anthropic"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    def validate_provider_config(self) -> None:
        """Validate that the selected provider has valid configuration."""
        if self.default_provider == "openai" and not self.openai.api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        if self.default_provider == "anthropic" and not self.anthropic.api_key:
            raise ValueError("Anthropic API key is required when using Anthropic provider")

    def get_active_provider_config(self) -> OllamaConfig | OpenAIConfig | AnthropicConfig:
        """Get configuration for the active provider."""
        if self.default_provider == "openai":
            return self.openai
        if self.default_provider == "anthropic":
            return self.anthropic
        return self.ollama

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}
