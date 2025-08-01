"""
Main application configuration combining all component configs.
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

from .database import DatabaseConfig
from .llm import LLMConfig
from .mcp import MCPConfig
from .observability import ObservabilityConfig


class AppConfig(BaseSettings):
    """
    Main application configuration.

    Combines all component configurations into a single, centralized
    configuration class for the SQLite Knowledge Graph with Vector Search.
    """

    # Application metadata
    app_name: str = Field(default="sqlite-kg-vec-mcp", description="Application name")

    app_version: str = Field(default="0.2.0", description="Application version")

    environment: str = Field(
        default="development", description="Environment (development, staging, production)"
    )

    debug: bool = Field(default=False, description="Enable debug mode")

    # Data directory
    data_dir: str = Field(default="data", description="Base directory for data files")

    @property
    def data_directory(self) -> Path:
        """Get data directory path, creating it if needed."""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Component configurations
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")

    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP server configuration")

    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="Observability configuration"
    )

    def model_post_init(self, __context: Any) -> None:  # pylint: disable=arguments-differ
        """Post-initialization validation and setup."""
        # Update nested configs with environment-specific settings
        self._update_nested_configs()

    def _update_nested_configs(self):
        """Update nested configurations with app-level settings."""
        # Update observability config with app metadata
        self.observability.service_name = self.app_name
        self.observability.service_version = self.app_version
        self.observability.environment = self.environment

        # Update database path to be relative to data_dir if not absolute
        if not Path(self.database.db_path).is_absolute():
            self.database.db_path = str(self.data_directory / self.database.db_path)

        # Update MCP vector index directory if not set
        if self.mcp.vector_index_dir is None:
            self.mcp.vector_index_dir = str(self.data_directory / "vector_index")

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "AppConfig":
        """
        Create configuration from environment variables and .env file.

        Args:
            env_file: Path to .env file (defaults to .env in current directory)

        Returns:
            AppConfig instance
        """
        env_file = env_file or ".env"

        # Load each component config from environment
        database_config = DatabaseConfig()
        llm_config = LLMConfig()
        mcp_config = MCPConfig()
        observability_config = ObservabilityConfig()

        # Create main config
        return cls(
            database=database_config,
            llm=llm_config,
            mcp=mcp_config,
            observability=observability_config,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def get_database_url(self) -> str:
        """Get database URL for SQLite."""
        return f"sqlite:///{self.database.db_path}"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore",
    }
