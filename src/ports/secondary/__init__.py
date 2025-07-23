"""
Secondary ports (driven ports) for the knowledge graph system.

These are interfaces that define how the domain interacts with external systems
(outbound/right-side ports in hexagonal architecture).
"""

# Repository ports
from .repositories import (
    EntityRepository,
    RelationshipRepository,
    EmbeddingRepository,
    VectorIndexRepository,
)

# External service ports
from .external_services import (
    LLMService,
    TextEmbedder,
    MonitoringService,
)

# Infrastructure ports
from .infrastructure import (
    Database,
    VectorStore,
    Cache,
)

__all__ = [
    # Repositories
    "EntityRepository",
    "RelationshipRepository",
    "EmbeddingRepository",
    "VectorIndexRepository",

    # External Services
    "LLMService",
    "TextEmbedder",
    "MonitoringService",

    # Infrastructure
    "Database",
    "VectorStore",
    "Cache",
]
