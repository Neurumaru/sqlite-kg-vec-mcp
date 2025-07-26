"""
Database infrastructure port for data persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncContextManager
from contextlib import asynccontextmanager


class Database(ABC):
    """
    Secondary port for database infrastructure operations.

    This interface defines how the domain interacts with the database layer
    for transactions, connections, and low-level operations.
    """

    # Connection management
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close database connection.

        Returns:
            True if disconnection was successful
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if database is connected.

        Returns:
            True if database is connected
        """
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """
        Ping the database to check connectivity.

        Returns:
            True if database responds
        """
        pass

    # Transaction management
    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[None]:
        """
        Create a database transaction context.

        Yields:
            Transaction context
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> str:
        """
        Begin a new transaction.

        Returns:
            Transaction ID
        """
        pass

    @abstractmethod
    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: Transaction ID to commit

        Returns:
            True if commit was successful
        """
        pass

    @abstractmethod
    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.

        Args:
            transaction_id: Transaction ID to rollback

        Returns:
            True if rollback was successful
        """
        pass

    # Query execution
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query.

        Args:
            query: SQL query to execute
            parameters: Optional query parameters
            transaction_id: Optional transaction ID

        Returns:
            Query results as list of dictionaries
        """
        pass

    @abstractmethod
    async def execute_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None
    ) -> int:
        """
        Execute a non-SELECT command (INSERT, UPDATE, DELETE).

        Args:
            command: SQL command to execute
            parameters: Optional command parameters
            transaction_id: Optional transaction ID

        Returns:
            Number of affected rows
        """
        pass

    @abstractmethod
    async def execute_batch(
        self,
        commands: List[str],
        parameters: Optional[List[Dict[str, Any]]] = None,
        transaction_id: Optional[str] = None
    ) -> List[int]:
        """
        Execute multiple commands in batch.

        Args:
            commands: List of SQL commands
            parameters: Optional list of parameters for each command
            transaction_id: Optional transaction ID

        Returns:
            List of affected row counts
        """
        pass

    # Schema management
    @abstractmethod
    async def create_table(
        self,
        table_name: str,
        schema: Dict[str, Any],
        if_not_exists: bool = True
    ) -> bool:
        """
        Create a database table.

        Args:
            table_name: Name of the table
            schema: Table schema definition
            if_not_exists: Whether to use IF NOT EXISTS clause

        Returns:
            True if table creation was successful
        """
        pass

    @abstractmethod
    async def drop_table(
        self,
        table_name: str,
        if_exists: bool = True
    ) -> bool:
        """
        Drop a database table.

        Args:
            table_name: Name of the table
            if_exists: Whether to use IF EXISTS clause

        Returns:
            True if table drop was successful
        """
        pass

    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        pass

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema of a table.

        Args:
            table_name: Name of the table

        Returns:
            Table schema or None if table doesn't exist
        """
        pass

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: List[str],
        unique: bool = False,
        if_not_exists: bool = True
    ) -> bool:
        """
        Create a database index.

        Args:
            index_name: Name of the index
            table_name: Table to index
            columns: Columns to include in index
            unique: Whether index should be unique
            if_not_exists: Whether to use IF NOT EXISTS clause

        Returns:
            True if index creation was successful
        """
        pass

    @abstractmethod
    async def drop_index(
        self,
        index_name: str,
        if_exists: bool = True
    ) -> bool:
        """
        Drop a database index.

        Args:
            index_name: Name of the index
            if_exists: Whether to use IF EXISTS clause

        Returns:
            True if index drop was successful
        """
        pass

    # Database maintenance
    @abstractmethod
    async def vacuum(self) -> bool:
        """
        Perform database vacuum operation.

        Returns:
            True if vacuum was successful
        """
        pass

    @abstractmethod
    async def analyze(self, table_name: Optional[str] = None) -> bool:
        """
        Analyze database statistics.

        Args:
            table_name: Optional specific table to analyze

        Returns:
            True if analysis was successful
        """
        pass

    @abstractmethod
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.

        Returns:
            Database information
        """
        pass

    @abstractmethod
    async def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Table information or None if table doesn't exist
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information and status.

        Returns:
            Connection information
        """
        pass

    @abstractmethod
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get database performance statistics.

        Returns:
            Performance statistics
        """
        pass