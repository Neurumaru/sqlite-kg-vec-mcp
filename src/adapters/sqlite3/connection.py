"""
SQLite database connection management module.
"""

import datetime
import sqlite3
import warnings
from pathlib import Path
from typing import Optional, Union

from src.common.observability import get_observable_logger

from .exceptions import SQLiteConnectionException


# Define custom timestamp converter functions for Python 3.12 compatibility
def adapt_datetime(dt: datetime.datetime) -> str:
    """Convert datetime to string for SQLite storage."""
    return dt.isoformat()


def convert_datetime(s: Union[str, bytes]) -> Union[datetime.datetime, str, bytes]:
    """Convert string from SQLite back to datetime."""
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return datetime.datetime.fromisoformat(s)
    except (ValueError, AttributeError, UnicodeDecodeError) as exception:
        warnings.warn(f"Failed to convert datetime {s!r}: {exception}")
        return s


# Register custom timestamp handlers
sqlite3.register_adapter(datetime.datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)


class DatabaseConnection:
    """
    Manages SQLite database connections with optimized settings.
    """

    def __init__(self, db_path: Union[str, Path], optimize: bool = True):
        """
        Initialize a database connection.
        Args:
            db_path: Path to the SQLite database file
            optimize: Whether to apply optimization PRAGMAs
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self.optimize = optimize
        self.logger = get_observable_logger("database_connection", "adapter")

    def connect(self) -> sqlite3.Connection:
        """
        Establish a connection to the SQLite database.
        Returns:
            SQLite connection object
        Raises:
            sqlite3.Error: If database connection fails
            PermissionError: If database file cannot be created/accessed
        """
        try:
            # Create directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exception:
            raise PermissionError(
                f"Cannot create database directory {self.db_path.parent}: {exception}"
            )
        try:
            # Connect to database
            self.connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                isolation_level=None,  # We'll manage transactions explicitly
                check_same_thread=False,  # Allow use from multiple threads
                timeout=30.0,  # Add connection timeout
            )
            # Enable returning rows as dictionaries
            self.connection.row_factory = sqlite3.Row
            # Test the connection
            self.connection.execute("SELECT 1").fetchone()
            # Apply optimization PRAGMAs if requested
            if self.optimize:
                self._apply_optimizations()
            return self.connection
        except sqlite3.OperationalError as exception:
            raise SQLiteConnectionException.from_sqlite_error(str(self.db_path), exception)
        except sqlite3.Error as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"Database connection failed: {exception}",
                original_error=exception,
            )
        except PermissionError as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"Permission denied accessing database: {exception}",
                original_error=exception,
            )
        except Exception as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"Unexpected error connecting to database: {exception}",
                original_error=exception,
            )

    def _apply_optimizations(self) -> None:
        """Apply SQLite optimizations via PRAGMA statements."""
        if not self.connection:
            return
        cursor = self.connection.cursor()
        # WAL mode for better concurrency and performance
        cursor.execute("PRAGMA journal_mode=WAL;")
        # Set busy timeout to prevent immediate lock errors (5 seconds)
        cursor.execute("PRAGMA busy_timeout=5000;")
        # Normal synchronization mode (balance between durability and performance)
        cursor.execute("PRAGMA synchronous=NORMAL;")
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON;")
        # Use memory for temp storage
        cursor.execute("PRAGMA temp_store=MEMORY;")
        # Larger cache for better performance (32MB)
        cursor.execute("PRAGMA cache_size=-32000;")  # negative means kilobytes
        cursor.close()

    def close(self) -> None:
        """Close the database connection if open."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> sqlite3.Connection:
        """Context manager enter method."""
        if not self.connection:
            return self.connect()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit method."""
        self.close()
