"""
SQLite database schema definitions and initialization.
"""

import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .connection import DatabaseConnection


class SchemaManager:
    """
    Manages the database schema for the knowledge graph and vector storage.
    """

    CURRENT_SCHEMA_VERSION = 3  # Current schema version

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the schema manager.
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_connection = DatabaseConnection(db_path)

    def initialize_schema(self) -> None:
        """
        Initialize the database schema by creating all necessary tables, indices, and triggers.
        """
        with self.db_connection as conn:
            self._create_schema_version_table(conn)
            self._create_entity_tables(conn)
            self._create_edge_tables(conn)
            self._create_hyperedge_tables(conn)
            self._create_document_tables(conn)
            self._create_observation_tables(conn)
            self._create_embedding_tables(conn)
            self._create_sync_tables(conn)
            # Set initial schema version
            self._update_schema_version(conn, 1)

    def _create_schema_version_table(self, conn: sqlite3.Connection) -> None:
        """Create the schema version tracking table."""
        conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1), -- Only one row allowed
            version INTEGER NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """
        )

    def _create_entity_tables(self, conn: sqlite3.Connection) -> None:
        """Create entity (node) related tables."""
        conn.executescript(
            """
        -- Entities (Nodes) table
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            uuid TEXT UNIQUE NOT NULL, -- Stable identifier for external reference
            name TEXT,
            type TEXT NOT NULL,
            properties JSON,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- Indices for entity table
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_uuid ON entities(uuid);
        -- Trigger to update the updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS trg_entities_updated_at
        AFTER UPDATE ON entities
        FOR EACH ROW
        BEGIN
            UPDATE entities SET updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
        """
        )

    def _create_edge_tables(self, conn: sqlite3.Connection) -> None:
        """Create binary edge related tables."""
        conn.executescript(
            """
        -- Binary relationships (edges) table
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            properties JSON,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- Indices for edge table
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_relation_type ON edges(relation_type);
        CREATE INDEX IF NOT EXISTS idx_edges_source_relation ON edges(source_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_edges_target_relation ON edges(target_id, relation_type);
        -- Trigger to update the updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS trg_edges_updated_at
        AFTER UPDATE ON edges
        FOR EACH ROW
        BEGIN
            UPDATE edges SET updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
        """
        )

    def _create_hyperedge_tables(self, conn: sqlite3.Connection) -> None:
        """Create hyperedge (n-ary relationships) related tables."""
        conn.executescript(
            """
        -- Hyperedges table (for n-ary relationships)
        CREATE TABLE IF NOT EXISTS hyperedges (
            id INTEGER PRIMARY KEY,
            hyperedge_type TEXT NOT NULL,
            properties JSON,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- Hyperedge members table (connects entities to hyperedges)
        CREATE TABLE IF NOT EXISTS hyperedge_members (
            hyperedge_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (hyperedge_id, entity_id, role),
            FOREIGN KEY (hyperedge_id) REFERENCES hyperedges(id) ON DELETE CASCADE,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- Indices for hyperedge tables
        CREATE INDEX IF NOT EXISTS idx_hyperedges_type ON hyperedges(hyperedge_type);
        CREATE INDEX IF NOT EXISTS idx_hyperedge_members_entity ON hyperedge_members(entity_id);
        CREATE INDEX IF NOT EXISTS idx_hyperedge_members_role ON hyperedge_members(role);
        -- Trigger to update the updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS trg_hyperedges_updated_at
        AFTER UPDATE ON hyperedges
        FOR EACH ROW
        BEGIN
            UPDATE hyperedges SET updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
        """
        )

    def _create_document_tables(self, conn: sqlite3.Connection) -> None:
        """Create document related tables."""
        conn.executescript(
            """
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            metadata JSON,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT,
            connected_nodes JSON DEFAULT '[]',
            connected_relationships JSON DEFAULT '[]'
        );
        -- Indices for document table
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
        CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at);
        CREATE INDEX IF NOT EXISTS idx_documents_version ON documents(version);
        -- TODO: 성능 최적화 - 추가 인덱스 검토 필요
        -- 1. 복합 인덱스: (status, created_at) - 미처리 문서 조회 최적화
        -- 2. 복합 인덱스: (doc_type, status) - 타입별 상태 조회 최적화
        -- 3. FTS 인덱스: title, content 전문 검색 성능 향상
        -- CREATE INDEX IF NOT EXISTS idx_documents_status_created ON documents(status, created_at);
        -- CREATE INDEX IF NOT EXISTS idx_documents_type_status ON documents(doc_type, status);
        -- CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(title, content, content='documents', content_rowid='rowid');
        -- Trigger to update the updated_at timestamp and increment version
        CREATE TRIGGER IF NOT EXISTS trg_documents_updated_at
        AFTER UPDATE ON documents
        FOR EACH ROW
        BEGIN
            UPDATE documents SET
                updated_at = datetime('now'),
                version = NEW.version + 1
            WHERE id = NEW.id;
        END;
        """
        )

    def _create_observation_tables(self, conn: sqlite3.Connection) -> None:
        """Create observation related tables (for cold data storage)."""
        conn.executescript(
            """
        -- Observations table (for cold/historical data)
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata JSON,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- Index for observation table
        CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
        CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at);
        """
        )

    def _create_embedding_tables(self, conn: sqlite3.Connection) -> None:
        """Create vector embedding related tables."""
        conn.executescript(
            """
        -- Node embeddings table
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL, -- Binary representation of the vector
            dimensions INTEGER NOT NULL, -- Number of dimensions in the vector
            model_info TEXT NOT NULL, -- Information about the embedding model
            embedding_version INTEGER NOT NULL DEFAULT 1, -- For versioning/tracking
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (node_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- Edge embeddings table
        CREATE TABLE IF NOT EXISTS edge_embeddings (
            edge_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            model_info TEXT NOT NULL,
            embedding_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (edge_id) REFERENCES edges(id) ON DELETE CASCADE
        );
        -- Hyperedge embeddings table
        CREATE TABLE IF NOT EXISTS hyperedge_embeddings (
            hyperedge_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            model_info TEXT NOT NULL,
            embedding_version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (hyperedge_id) REFERENCES hyperedges(id) ON DELETE CASCADE
        );
        -- Triggers to update the updated_at timestamps
        CREATE TRIGGER IF NOT EXISTS trg_node_embeddings_updated_at
        AFTER UPDATE ON node_embeddings
        FOR EACH ROW
        BEGIN
            UPDATE node_embeddings SET updated_at = datetime('now')
            WHERE node_id = NEW.node_id;
        END;
        CREATE TRIGGER IF NOT EXISTS trg_edge_embeddings_updated_at
        AFTER UPDATE ON edge_embeddings
        FOR EACH ROW
        BEGIN
            UPDATE edge_embeddings SET updated_at = datetime('now')
            WHERE edge_id = NEW.edge_id;
        END;
        CREATE TRIGGER IF NOT EXISTS trg_hyperedge_embeddings_updated_at
        AFTER UPDATE ON hyperedge_embeddings
        FOR EACH ROW
        BEGIN
            UPDATE hyperedge_embeddings SET updated_at = datetime('now')
            WHERE hyperedge_id = NEW.hyperedge_id;
        END;
        """
        )

    def _create_sync_tables(self, conn: sqlite3.Connection) -> None:
        """Create tables for vector-DB synchronization using outbox pattern."""
        conn.executescript(
            """
        -- Vector operations outbox table (for async processing)
        CREATE TABLE IF NOT EXISTS vector_outbox (
            id INTEGER PRIMARY KEY,
            operation_type TEXT NOT NULL, -- 'insert', 'update', 'delete'
            entity_type TEXT NOT NULL, -- 'node', 'edge', 'hyperedge'
            entity_id INTEGER NOT NULL,
            model_info TEXT,
            status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
            correlation_id TEXT, -- For tracking related operations
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- Index for efficient outbox processing
        CREATE INDEX IF NOT EXISTS idx_vector_outbox_status ON vector_outbox(status);
        CREATE INDEX IF NOT EXISTS idx_vector_outbox_entity ON vector_outbox(entity_type, entity_id);
        -- Sync failures logging table
        CREATE TABLE IF NOT EXISTS sync_failures (
            id INTEGER PRIMARY KEY,
            outbox_id INTEGER,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            operation_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            retry_count INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (outbox_id) REFERENCES vector_outbox(id) ON DELETE SET NULL
        );
        -- Trigger to update vector_outbox updated_at
        CREATE TRIGGER IF NOT EXISTS trg_vector_outbox_updated_at
        AFTER UPDATE ON vector_outbox
        FOR EACH ROW
        BEGIN
            UPDATE vector_outbox SET updated_at = datetime('now')
            WHERE id = NEW.id;
        END;
        """
        )

    def _update_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        """
        Update or insert the schema version.
        Args:
            conn: SQLite connection
            version: New schema version number
        """
        conn.execute(
            """
        INSERT INTO schema_version (id, version) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET
            version = excluded.version,
            updated_at = CURRENT_TIMESTAMP
        """,
            (version,),
        )

    def get_schema_version(self) -> int:
        """
        Get the current schema version.
        Returns:
            Current schema version number or 0 if not set
        """
        with self.db_connection as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT version FROM schema_version WHERE id = 1")
                result = cursor.fetchone()
                return result[0] if result else 0
            except sqlite3.OperationalError:
                # Schema version table doesn't exist, return 0
                return 0

    def migrate_schema(self, target_version: Optional[int] = None) -> bool:
        """
        Migrate schema to target version.
        Args:
            target_version: Target schema version (defaults to latest)
        Returns:
            True if migration was successful
        Raises:
            ValueError: If target version is invalid
            sqlite3.Error: If migration fails
        """
        if target_version is None:
            target_version = self.CURRENT_SCHEMA_VERSION
        current_version = self.get_schema_version()
        if current_version == target_version:
            return True
        if target_version < current_version:
            raise ValueError(
                f"Downgrade from version {current_version} to {target_version} not supported"
            )
        if target_version > self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Target version {target_version} is higher than latest version {self.CURRENT_SCHEMA_VERSION}"
            )
        # Apply migrations step by step
        conn = self.db_connection.connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            try:
                for version in range(current_version + 1, target_version + 1):
                    self._apply_migration(conn, version)
                    self._update_schema_version(conn, version)
                conn.execute("COMMIT")
                return True
            except Exception as exception:
                conn.execute("ROLLBACK")
                raise sqlite3.Error(f"Migration to version {target_version} failed: {exception}")
        finally:
            conn.close()

    def _apply_migration(self, conn: sqlite3.Connection, version: int) -> None:
        """
        Apply a specific migration version.
        Args:
            conn: SQLite connection
            version: Migration version to apply
        """
        if version == 1:
            # Initial schema creation
            self._create_entity_tables(conn)
            self._create_edge_tables(conn)
            self._create_hyperedge_tables(conn)
            self._create_embedding_tables(conn)
            self._create_observation_tables(conn)
        elif version == 2:
            # Add generated columns for JSON optimization (if not already added)
            self._add_json_optimization_columns(conn)
        elif version == 3:
            # Add documents table
            self._create_document_tables(conn)
        else:
            raise ValueError(f"Unknown migration version: {version}")

    def _add_json_optimization_columns(self, conn: sqlite3.Connection) -> None:
        """Add JSON optimization columns if they don't exist."""
        try:
            # Check if columns already exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(entities)")
            columns = [row[1] for row in cursor.fetchall()]
            if "json_text_content" not in columns:
                conn.executescript(
                    """
                -- Add generated columns for entities
                ALTER TABLE entities ADD COLUMN json_text_content TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.bio')) STORED;
                ALTER TABLE entities ADD COLUMN json_category TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.category')) STORED;
                ALTER TABLE entities ADD COLUMN json_status TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.status')) STORED;
                -- Add indices
                CREATE INDEX IF NOT EXISTS idx_entities_json_text ON entities(json_text_content) WHERE json_text_content IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_entities_json_category ON entities(json_category) WHERE json_category IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_entities_json_status ON entities(json_status) WHERE json_status IS NOT NULL;
                """
                )
            # Check edges table
            cursor.execute("PRAGMA table_info(edges)")
            columns = [row[1] for row in cursor.fetchall()]
            if "json_weight" not in columns:
                conn.executescript(
                    """
                -- Add generated columns for edges
                ALTER TABLE edges ADD COLUMN json_weight REAL
                    GENERATED ALWAYS AS (CAST(JSON_EXTRACT(properties, '$.weight') AS REAL)) STORED;
                ALTER TABLE edges ADD COLUMN json_since TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.since')) STORED;
                ALTER TABLE edges ADD COLUMN json_confidence REAL
                    GENERATED ALWAYS AS (CAST(JSON_EXTRACT(properties, '$.confidence') AS REAL)) STORED;
                -- Add indices
                CREATE INDEX IF NOT EXISTS idx_edges_json_weight ON edges(json_weight) WHERE json_weight IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_edges_json_since ON edges(json_since) WHERE json_since IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_edges_json_confidence ON edges(json_confidence) WHERE json_confidence IS NOT NULL;
                """
                )
        except sqlite3.Error as exception:
            # If generated columns are not supported (older SQLite), skip silently
            warnings.warn(f"Could not add JSON optimization columns: {exception}")

    def backup_schema(self, backup_path: str) -> bool:
        """
        Create a backup of the current database.
        Args:
            backup_path: Path for the backup file
        Returns:
            True if backup was successful
        """
        try:
            with self.db_connection as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            return True
        except Exception as exception:
            raise sqlite3.Error(f"Backup failed: {exception}")

    def validate_schema(self) -> dict:
        """
        Validate the current schema integrity.
        Returns:
            Dictionary with validation results
        """
        # Get version first before any connection operations
        try:
            version = self.get_schema_version()
        except (sqlite3.Error, Exception):
            version = 0
        results: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "version": version}
        conn = self.db_connection.connect()
        try:
            # Check foreign key integrity
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_key_check")
            fk_errors = cursor.fetchall()
            if fk_errors:
                results["valid"] = False
                error_list = [f"Foreign key error: {error}" for error in fk_errors]
                results["errors"].extend(error_list)
            # Check table integrity
            for table in [
                "entities",
                "edges",
                "hyperedges",
                "node_embeddings",
                "edge_embeddings",
            ]:
                try:
                    cursor.execute(f"PRAGMA integrity_check({table})")
                    integrity = cursor.fetchone()[0]
                    if integrity != "ok":
                        results["valid"] = False
                        results["errors"].append(f"Integrity check failed for {table}: {integrity}")
                except sqlite3.Error:
                    # Table might not exist, skip
                    pass
            # Check indices
            cursor.execute("PRAGMA index_list(entities)")
            if not cursor.fetchall():
                warnings_list = results["warnings"]
                warnings_list.append("No indices found on entities table")
        except sqlite3.Error as exception:
            results["valid"] = False
            errors_list = results["errors"]
            errors_list.append(f"Schema validation error: {exception}")
        finally:
            conn.close()
        return results
