"""
SQLite implementation of the Database port.
"""

import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncContextManager, Dict, List, Optional

from src.ports.database import Database

from .connection import DatabaseConnection


class SQLiteDatabase(Database):
    """
    SQLite implementation of the Database port.

    This adapter provides concrete implementation of database operations
    using SQLite as the underlying storage engine.
    """

    def __init__(self, db_path: str, optimize: bool = True):
        """
        Initialize SQLite database adapter.

        Args:
            db_path: Path to the SQLite database file
            optimize: Whether to apply optimization PRAGMAs
        """
        self.db_path = Path(db_path)
        self.optimize = optimize
        self._connection_manager = DatabaseConnection(db_path, optimize)
        self._connection: Optional[sqlite3.Connection] = None
        self._active_transactions: Dict[str, sqlite3.Connection] = {}

    # Connection management
    async def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            True if connection was successful
        """
        try:
            self._connection = self._connection_manager.connect()
            return True
        except Exception:
            return False

    async def disconnect(self) -> bool:
        """
        Close database connection.

        Returns:
            True if disconnection was successful
        """
        try:
            # Close all active transactions
            for transaction_conn in self._active_transactions.values():
                try:
                    transaction_conn.rollback()
                    transaction_conn.close()
                except Exception:
                    pass
            self._active_transactions.clear()

            # Close main connection
            self._connection_manager.close()
            self._connection = None
            return True
        except Exception:
            return False

    async def is_connected(self) -> bool:
        """
        Check if database is connected.

        Returns:
            True if database is connected
        """
        if not self._connection:
            return False

        try:
            self._connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    async def ping(self) -> bool:
        """
        Ping the database to check connectivity.

        Returns:
            True if database responds
        """
        return await self.is_connected()

    # Transaction management
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[None]:
        """
        Create a database transaction context.

        Yields:
            Transaction context
        """
        transaction_id = await self.begin_transaction()
        try:
            yield
            await self.commit_transaction(transaction_id)
        except Exception:
            await self.rollback_transaction(transaction_id)
            raise

    async def begin_transaction(self) -> str:
        """
        Begin a new transaction.

        Returns:
            Transaction ID
        """
        if not self._connection:
            raise RuntimeError("Database not connected")

        transaction_id = str(uuid.uuid4())
        # For SQLite, we'll use the same connection but track transaction state
        self._active_transactions[transaction_id] = self._connection
        self._connection.execute("BEGIN")
        return transaction_id

    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: Transaction ID to commit

        Returns:
            True if commit was successful
        """
        if transaction_id not in self._active_transactions:
            return False

        try:
            connection = self._active_transactions[transaction_id]
            connection.commit()
            del self._active_transactions[transaction_id]
            return True
        except Exception:
            return False

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a transaction.

        Args:
            transaction_id: Transaction ID to rollback

        Returns:
            True if rollback was successful
        """
        if transaction_id not in self._active_transactions:
            return False

        try:
            connection = self._active_transactions[transaction_id]
            connection.rollback()
            del self._active_transactions[transaction_id]
            return True
        except Exception:
            return False

    # Query execution
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
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
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("Database not connected")

        cursor = connection.cursor()
        try:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            # Convert rows to dictionaries
            columns = (
                [description[0] for description in cursor.description]
                if cursor.description
                else []
            )
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    async def execute_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
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
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("Database not connected")

        cursor = connection.cursor()
        try:
            if parameters:
                cursor.execute(command, parameters)
            else:
                cursor.execute(command)
            return cursor.rowcount
        finally:
            cursor.close()

    async def execute_batch(
        self,
        commands: List[str],
        parameters: Optional[List[Dict[str, Any]]] = None,
        transaction_id: Optional[str] = None,
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
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("Database not connected")

        results = []
        for i, command in enumerate(commands):
            cursor = connection.cursor()
            try:
                cmd_params = (
                    parameters[i] if parameters and i < len(parameters) else None
                )
                if cmd_params:
                    cursor.execute(command, cmd_params)
                else:
                    cursor.execute(command)
                results.append(cursor.rowcount)
            finally:
                cursor.close()

        return results

    # Schema management
    async def create_table(
        self, table_name: str, schema: Dict[str, Any], if_not_exists: bool = True
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
        try:
            # Build CREATE TABLE statement from schema
            columns = []
            for column_name, column_def in schema.items():
                if isinstance(column_def, str):
                    columns.append(f"{column_name} {column_def}")
                elif isinstance(column_def, dict):
                    col_type = column_def.get("type", "TEXT")
                    col_def = f"{column_name} {col_type}"
                    if column_def.get("primary_key"):
                        col_def += " PRIMARY KEY"
                    if column_def.get("not_null"):
                        col_def += " NOT NULL"
                    if "default" in column_def:
                        col_def += f" DEFAULT {column_def['default']}"
                    columns.append(col_def)

            if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
            sql = f"CREATE TABLE {if_not_exists_clause}{table_name} ({', '.join(columns)})"

            await self.execute_command(sql)
            return True
        except Exception:
            return False

    async def drop_table(self, table_name: str, if_exists: bool = True) -> bool:
        """
        Drop a database table.

        Args:
            table_name: Name of the table
            if_exists: Whether to use IF EXISTS clause

        Returns:
            True if table drop was successful
        """
        try:
            if_exists_clause = "IF EXISTS " if if_exists else ""
            sql = f"DROP TABLE {if_exists_clause}{table_name}"
            await self.execute_command(sql)
            return True
        except Exception:
            return False

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        try:
            result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                {"name": table_name},
            )
            return len(result) > 0
        except Exception:
            return False

    async def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema of a table.

        Args:
            table_name: Name of the table

        Returns:
            Table schema or None if table doesn't exist
        """
        try:
            result = await self.execute_query(f"PRAGMA table_info({table_name})")
            if not result:
                return None

            schema = {}
            for row in result:
                schema[row["name"]] = {
                    "type": row["type"],
                    "not_null": bool(row["notnull"]),
                    "default": row["dflt_value"],
                    "primary_key": bool(row["pk"]),
                }
            return schema
        except Exception:
            return None

    async def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: List[str],
        unique: bool = False,
        if_not_exists: bool = True,
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
        try:
            unique_clause = "UNIQUE " if unique else ""
            if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
            columns_str = ", ".join(columns)

            sql = f"CREATE {unique_clause}INDEX {if_not_exists_clause}{index_name} ON {table_name} ({columns_str})"
            await self.execute_command(sql)
            return True
        except Exception:
            return False

    async def drop_index(self, index_name: str, if_exists: bool = True) -> bool:
        """
        Drop a database index.

        Args:
            index_name: Name of the index
            if_exists: Whether to use IF EXISTS clause

        Returns:
            True if index drop was successful
        """
        try:
            if_exists_clause = "IF EXISTS " if if_exists else ""
            sql = f"DROP INDEX {if_exists_clause}{index_name}"
            await self.execute_command(sql)
            return True
        except Exception:
            return False

    # Database maintenance
    async def vacuum(self) -> bool:
        """
        Perform database vacuum operation.

        Returns:
            True if vacuum was successful
        """
        try:
            await self.execute_command("VACUUM")
            return True
        except Exception:
            return False

    async def analyze(self, table_name: Optional[str] = None) -> bool:
        """
        Analyze database statistics.

        Args:
            table_name: Optional specific table to analyze

        Returns:
            True if analysis was successful
        """
        try:
            if table_name:
                await self.execute_command(f"ANALYZE {table_name}")
            else:
                await self.execute_command("ANALYZE")
            return True
        except Exception:
            return False

    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.

        Returns:
            Database information
        """
        try:
            info = {
                "path": str(self.db_path),
                "size_bytes": (
                    self.db_path.stat().st_size if self.db_path.exists() else 0
                ),
            }

            # Get SQLite version and settings
            version_result = await self.execute_query("SELECT sqlite_version()")
            if version_result:
                info["sqlite_version"] = version_result[0]["sqlite_version()"]

            # Get pragma settings
            pragma_queries = [
                "PRAGMA journal_mode",
                "PRAGMA synchronous",
                "PRAGMA cache_size",
                "PRAGMA foreign_keys",
            ]

            for pragma in pragma_queries:
                try:
                    result = await self.execute_query(pragma)
                    if result:
                        key = pragma.split()[-1]
                        info[key] = result[0][pragma.replace("PRAGMA ", "")]
                except Exception:
                    continue

            return info
        except Exception:
            return {"error": "Failed to get database info"}

    async def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Table information or None if table doesn't exist
        """
        try:
            # Check if table exists
            if not await self.table_exists(table_name):
                return None

            info = {"name": table_name}

            # Get row count
            count_result = await self.execute_query(
                f"SELECT COUNT(*) as count FROM {table_name}"
            )
            if count_result:
                info["row_count"] = count_result[0]["count"]

            # Get schema
            schema = await self.get_table_schema(table_name)
            if schema:
                info["schema"] = schema

            return info
        except Exception:
            return None

    # Health and diagnostics
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Health status information
        """
        health = {
            "connected": await self.is_connected(),
            "file_exists": self.db_path.exists(),
            "readable": False,
            "writable": False,
        }

        if health["connected"]:
            try:
                # Test read
                await self.execute_query("SELECT 1")
                health["readable"] = True

                # Test write (create and drop temp table)
                await self.execute_command(
                    "CREATE TEMP TABLE health_check_temp (id INTEGER)"
                )
                await self.execute_command("DROP TABLE health_check_temp")
                health["writable"] = True
            except Exception as e:
                health["error"] = str(e)

        health["status"] = (
            "healthy"
            if all([health["connected"], health["readable"], health["writable"]])
            else "unhealthy"
        )

        return health

    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information and status.

        Returns:
            Connection information
        """
        return {
            "connected": await self.is_connected(),
            "db_path": str(self.db_path),
            "optimize": self.optimize,
            "active_transactions": len(self._active_transactions),
            "transaction_ids": list(self._active_transactions.keys()),
        }

    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get database performance statistics.

        Returns:
            Performance statistics
        """
        try:
            stats = {}

            # Database size
            if self.db_path.exists():
                stats["file_size_bytes"] = self.db_path.stat().st_size

            # Table statistics
            tables_result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )

            stats["table_count"] = len(tables_result)
            stats["tables"] = {}

            for table_row in tables_result:
                table_name = table_row["name"]
                table_info = await self.get_table_info(table_name)
                if table_info:
                    stats["tables"][table_name] = {
                        "row_count": table_info.get("row_count", 0)
                    }

            return stats
        except Exception:
            return {"error": "Failed to get performance stats"}

    def _get_connection(
        self, transaction_id: Optional[str] = None
    ) -> Optional[sqlite3.Connection]:
        """
        Get the appropriate connection for the transaction.

        Args:
            transaction_id: Optional transaction ID

        Returns:
            SQLite connection or None
        """
        if transaction_id and transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id]
        return self._connection
