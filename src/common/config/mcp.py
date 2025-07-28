"""
MCP (Model Context Protocol) server configuration settings.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class MCPConfig(BaseSettings):
    """
    MCP server configuration settings.
    
    Configuration for the FastMCP-based knowledge graph server.
    """
    
    # Server settings
    server_name: str = Field(
        default="Knowledge Graph Server",
        description="Name of the MCP server"
    )
    
    server_instructions: str = Field(
        default="SQLite-based knowledge graph with vector search capabilities",
        description="Instructions/description for the MCP server"
    )
    
    host: str = Field(
        default="localhost",
        description="Server host address"
    )
    
    port: int = Field(
        default=8000,
        description="Server port number"
    )
    
    # Vector settings
    vector_index_dir: Optional[str] = Field(
        default=None,
        description="Directory for storing vector index files"
    )
    
    embedding_dim: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )
    
    vector_similarity: str = Field(
        default="cosine",
        description="Vector similarity metric (cosine, euclidean, dot)"
    )
    
    # Embedding settings
    embedder_type: str = Field(
        default="sentence-transformers",
        description="Type of text embedder to use"
    )
    
    embedder_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for embeddings"
    )
    
    # Performance settings
    max_connections: int = Field(
        default=100,
        description="Maximum concurrent connections"
    )
    
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    
    # Search settings
    max_search_results: int = Field(
        default=50,
        description="Maximum number of search results to return"
    )
    
    search_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for search results"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level for MCP server"
    )
    
    enable_debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # CORS settings
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS support"
    )
    
    cors_origins: list = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v
    
    @field_validator("vector_similarity")
    @classmethod
    def validate_vector_similarity(cls, v):
        """Validate vector similarity metric."""
        valid_metrics = {"cosine", "euclidean", "dot", "l2"}
        if v not in valid_metrics:
            raise ValueError(f"Vector similarity must be one of {valid_metrics}")
        return v
    
    @field_validator("search_threshold")
    @classmethod
    def validate_search_threshold(cls, v):
        """Validate search threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Search threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("vector_index_dir")
    @classmethod
    def validate_vector_index_dir(cls, v):
        """Validate and create vector index directory."""
        if v is not None:
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        return v

    model_config = {
        "env_prefix": "MCP_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }