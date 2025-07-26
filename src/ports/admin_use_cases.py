"""
Primary port for administrative use cases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.domain import NodeId


class AdminUseCases(ABC):
    """
    Primary port defining administrative operations.

    This interface defines management and maintenance operations
    for the knowledge graph system.
    """

    # System information
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and statistics.

        Returns:
            System information dictionary
        """
        pass

    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health status information
        """
        pass

    # Knowledge extraction
    @abstractmethod
    async def extract_knowledge_from_text(
        self,
        text: str,
        source_name: Optional[str] = None,
        auto_create: bool = True
    ) -> Dict[str, Any]:
        """
        Extract knowledge from text using LLM.

        Args:
            text: Text to extract knowledge from
            source_name: Optional source name for tracking
            auto_create: Whether to automatically create entities/relationships

        Returns:
            Extraction results
        """
        pass

    # Batch operations
    @abstractmethod
    async def bulk_create_entities(
        self,
        entities_data: List[Dict[str, Any]]
    ) -> List[NodeId]:
        """
        Create multiple entities in batch.

        Args:
            entities_data: List of entity data dictionaries

        Returns:
            List of created entity IDs
        """
        pass

    @abstractmethod
    async def bulk_create_relationships(
        self,
        relationships_data: List[Dict[str, Any]]
    ) -> List[NodeId]:
        """
        Create multiple relationships in batch.

        Args:
            relationships_data: List of relationship data dictionaries

        Returns:
            List of created relationship IDs
        """
        pass

    # Data management
    @abstractmethod
    async def export_knowledge_graph(
        self,
        format: str = "json",
        include_embeddings: bool = False
    ) -> str:
        """
        Export the knowledge graph data.

        Args:
            format: Export format ("json", "csv", "rdf")
            include_embeddings: Whether to include embeddings

        Returns:
            Exported data as string
        """
        pass

    @abstractmethod
    async def import_knowledge_graph(
        self,
        data: str,
        format: str = "json",
        merge_strategy: str = "skip_existing"
    ) -> Dict[str, Any]:
        """
        Import knowledge graph data.

        Args:
            data: Data to import
            format: Data format
            merge_strategy: How to handle existing data

        Returns:
            Import results summary
        """
        pass

    # Index management
    @abstractmethod
    async def rebuild_vector_index(
        self,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Rebuild the vector search index.

        Args:
            force: Whether to force rebuild even if index is current

        Returns:
            Rebuild operation results
        """
        pass

    @abstractmethod
    async def optimize_database(self) -> Dict[str, Any]:
        """
        Optimize the database for better performance.

        Returns:
            Optimization results
        """
        pass

    # Monitoring and maintenance
    @abstractmethod
    async def cleanup_orphaned_data(self) -> Dict[str, Any]:
        """
        Clean up orphaned data in the system.

        Returns:
            Cleanup results
        """
        pass

    @abstractmethod
    async def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity across the system.

        Returns:
            Validation results
        """
        pass

    # Configuration
    @abstractmethod
    async def update_system_config(
        self,
        config_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update system configuration.

        Args:
            config_updates: Configuration updates

        Returns:
            Updated configuration
        """
        pass

    @abstractmethod
    async def get_system_config(self) -> Dict[str, Any]:
        """
        Get current system configuration.

        Returns:
            System configuration
        """
        pass
