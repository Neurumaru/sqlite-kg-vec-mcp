"""
Database configuration settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """
    Database configuration settings.

    Supports SQLite-specific settings for the knowledge graph database
    and vector store operations.
    """

    # SQLite database settings
    db_path: str = Field(
        default="data/knowledge_graph.db", description="Path to the SQLite database file"
    )

    optimize: bool = Field(default=True, description="Whether to apply SQLite optimization PRAGMAs")

    # Connection settings
    timeout: float = Field(default=30.0, description="Database connection timeout in seconds")

    check_same_thread: bool = Field(default=False, description="SQLite check_same_thread parameter")

    # Vector store settings
    vector_dimension: int = Field(default=384, description="Dimension of vector embeddings")

    max_connections: int = Field(default=10, description="Maximum number of database connections")

    # Backup settings
    backup_enabled: bool = Field(default=False, description="Whether to enable automatic backups")

    backup_interval: int = Field(default=3600, description="Backup interval in seconds")

    backup_path: str | None = Field(default=None, description="Path for database backups")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """Validate database path format."""
        # Only validate format, don't create directories in validator
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Database path must be a non-empty string")
        return v

    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """Validate vector dimension is positive."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v

    @field_validator("backup_path")
    @classmethod
    def validate_backup_path(cls, v: str | None) -> str | None:
        """Validate backup path format if provided."""
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("Backup path must be a non-empty string")
        return v

    @property
    def db_directory(self) -> Path:
        """Get database directory path, creating it if needed."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    @property
    def backup_directory(self) -> Path | None:
        """Get backup directory path, creating it if needed."""
        if self.backup_path is None:
            return None
        path = Path(self.backup_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_prefix": "DB_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }
