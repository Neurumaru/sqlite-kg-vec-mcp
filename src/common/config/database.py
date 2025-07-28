"""
Database configuration settings.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class DatabaseConfig(BaseSettings):
    """
    Database configuration settings.
    
    Supports SQLite-specific settings for the knowledge graph database
    and vector store operations.
    """
    
    # SQLite database settings
    db_path: str = Field(
        default="data/knowledge_graph.db",
        description="Path to the SQLite database file"
    )
    
    optimize: bool = Field(
        default=True,
        description="Whether to apply SQLite optimization PRAGMAs"
    )
    
    # Connection settings
    timeout: float = Field(
        default=30.0,
        description="Database connection timeout in seconds"
    )
    
    check_same_thread: bool = Field(
        default=False,
        description="SQLite check_same_thread parameter"
    )
    
    # Vector store settings
    vector_dimension: int = Field(
        default=384,
        description="Dimension of vector embeddings"
    )
    
    max_connections: int = Field(
        default=10,
        description="Maximum number of database connections"
    )
    
    # Backup settings
    backup_enabled: bool = Field(
        default=False,
        description="Whether to enable automatic backups"
    )
    
    backup_interval: int = Field(
        default=3600,
        description="Backup interval in seconds"
    )
    
    backup_path: Optional[str] = Field(
        default=None,
        description="Path for database backups"
    )

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v):
        """Validate database path and create directory if needed."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v):
        """Validate vector dimension is positive."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
    
    @field_validator("backup_path")
    @classmethod
    def validate_backup_path(cls, v):
        """Validate backup path if provided."""
        if v is not None:
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        return v

    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        case_sensitive = False