"""
Optimized database port.

Lightweight database interface focused on core functionality,
with unnecessary abstractions removed based on actual usage analysis.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any


class Database(ABC):
    """
    Optimized database port.

    Lightweight interface with only essential functionality
    based on actual usage patterns analysis.
    """

    # Core operations - most frequently used essential methods
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        Most frequently used core method.

        Args:
            query: SQL query to execute
            parameters: Optional query parameters
            transaction_id: Optional transaction ID

        Returns:
            Query results as list of dictionaries
        """

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        parameters: dict[str, Any] | None = None,
        transaction_id: str | None = None,
    ) -> int:
        """
        Execute non-SELECT commands (INSERT, UPDATE, DELETE).

        Second most frequently used core method.

        Args:
            command: SQL command to execute
            parameters: Optional command parameters
            transaction_id: Optional transaction ID

        Returns:
            Number of affected rows
        """

    # Transaction management - only context manager retained
    @abstractmethod
    def transaction(self) -> AbstractAsyncContextManager[None]:
        """
        Create a transaction context.

        Most transaction usage follows the context manager pattern.

        Returns:
            Transaction context manager
        """

    # Connection management - minimal connection handling
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            Connection success status
        """

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check database connection status.

        Returns:
            Connection status
        """

    # Schema inspection - methods used only when needed
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Table name

        Returns:
            Table existence status
        """

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> dict[str, Any] | None:
        """
        Get table schema.

        Args:
            table_name: Table name

        Returns:
            Table schema or None if table doesn't exist
        """


class DatabaseMaintenance(ABC):
    """
    Separate interface for database maintenance operations.

    Separated from regular database operations for use only when needed.
    Adapters can optionally implement this interface.
    """

    @abstractmethod
    async def vacuum(self) -> bool:
        """Perform database VACUUM operation."""

    @abstractmethod
    async def analyze(self, table_name: str | None = None) -> bool:
        """Analyze database statistics."""

    @abstractmethod
    async def create_table(
        self,
        table_name: str,
        schema: dict[str, Any],
        if_not_exists: bool = True,
    ) -> bool:
        """Create database table."""

    @abstractmethod
    async def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """Drop database table."""

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: list[str],
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> bool:
        """Create database index."""

    @abstractmethod
    async def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """Drop database index."""
