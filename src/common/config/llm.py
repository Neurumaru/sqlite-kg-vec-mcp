"""
LLM (Large Language Model) configuration settings.
"""

from typing import Dict, Optional

from pydantic import BaseSettings, Field, validator


class OllamaConfig(BaseSettings):
    """Ollama service configuration."""
    
    host: str = Field(
        default="localhost",
        description="Ollama server host"
    )
    
    port: int = Field(
        default=11434,
        description="Ollama server port"
    )
    
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    
    model: str = Field(
        default="llama3.2",
        description="Default Ollama model"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Default sampling temperature"
    )
    
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for responses"
    )

    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    class Config:
        env_prefix = "OLLAMA_"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Embedding dimension (model-specific)"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Default sampling temperature"
    )
    
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for responses"
    )
    
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )

    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("embedding_dimension")
    def validate_embedding_dimension(cls, v):
        """Validate embedding dimension."""
        if v is not None and v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    class Config:
        env_prefix = "OPENAI_"


class AnthropicConfig(BaseSettings):
    """Anthropic (Claude) API configuration."""
    
    api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    model: str = Field(
        default="claude-3-haiku-20240307",
        description="Default Claude model"
    )
    
    temperature: float = Field(
        default=0.7,
        description="Default sampling temperature"
    )
    
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for responses"
    )
    
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )

    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v

    class Config:
        env_prefix = "ANTHROPIC_"


class LLMConfig(BaseSettings):
    """
    Combined LLM configuration settings.
    
    Includes settings for all supported LLM providers.
    """
    
    # Default provider
    default_provider: str = Field(
        default="ollama",
        description="Default LLM provider (ollama, openai, anthropic)"
    )
    
    # Provider configs
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Ollama configuration"
    )
    
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI configuration"
    )
    
    anthropic: AnthropicConfig = Field(
        default_factory=AnthropicConfig,
        description="Anthropic configuration"
    )
    
    # Common settings
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests"
    )
    
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds"
    )

    @validator("default_provider")
    def validate_default_provider(cls, v):
        """Validate default provider."""
        valid_providers = {"ollama", "openai", "anthropic"}
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    class Config:
        env_prefix = "LLM_"
        env_file = ".env"