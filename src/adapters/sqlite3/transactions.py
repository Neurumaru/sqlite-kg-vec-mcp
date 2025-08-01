"""
Transaction management for SQLite database operations.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager


class TransactionManager:
    """
    Manages database transactions to ensure atomicity and consistency.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the transaction manager.
        Args:
            connection: SQLite database connection
        """
        self.connection = connection

    @contextmanager
    def transaction(
        self, isolation_level: str = "IMMEDIATE"
    ) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database transactions.
        Args:
            isolation_level: SQLite isolation level ('DEFERRED', 'IMMEDIATE', or 'EXCLUSIVE')
                            IMMEDIATE is safer for concurrent operations
        Yields:
            SQLite connection for executing statements within the transaction
        Raises:
            Any exception from the transaction context
        """
        # We need to use 'execute' here because we disabled automatic
        # transaction management when creating the connection
        self.connection.execute(f"BEGIN {isolation_level} TRANSACTION")
        try:
            yield self.connection
            self.connection.execute("COMMIT")
        except Exception as exception:
            self.connection.execute("ROLLBACK")
            raise exception


class UnitOfWork:
    """
    Implements the Unit of Work pattern for coordinating and tracking DB changes.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the Unit of Work.
        Args:
            connection: SQLite database connection
        """
        self.connection = connection
        self.transaction_manager = TransactionManager(connection)
        self._correlation_id: str | None = None

    @property
    def correlation_id(self) -> str | None:
        """Get the correlation ID for tracking related operations."""
        return self._correlation_id

    @correlation_id.setter
    def correlation_id(self, value: str) -> None:
        """Set the correlation ID for tracking related operations."""
        self._correlation_id = value

    @contextmanager
    def begin(
        self, isolation_level: str = "IMMEDIATE"
    ) -> Generator[sqlite3.Connection, None, None]:
        """
        Begin a unit of work (a transaction).
        Args:
            isolation_level: SQLite isolation level
        Yields:
            SQLite connection for executing statements
        """
        with self.transaction_manager.transaction(isolation_level) as conn:
            yield conn

    def register_vector_operation(
        self,
        entity_type: str,
        entity_id: int,
        operation_type: str,
        model_info: str | None = None,
    ) -> int:
        """
        Register a vector operation in the outbox for asynchronous processing.
        Args:
            entity_type: Type of entity ('node', 'edge', 'hyperedge')
            entity_id: ID of the entity
            operation_type: Type of operation ('insert', 'update', 'delete')
            model_info: Optional model information for embeddings
        Returns:
            ID of the created outbox entry
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
        INSERT INTO vector_outbox (
            operation_type, entity_type, entity_id, model_info, correlation_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
            (operation_type, entity_type, entity_id, model_info, self._correlation_id),
        )
        result = cursor.lastrowid
        if result is None:
            raise RuntimeError("Failed to insert into vector_outbox")
        return result
