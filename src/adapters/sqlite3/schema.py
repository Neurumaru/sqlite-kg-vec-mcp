"""
SQLite 데이터베이스 스키마 정의 및 초기화.
"""

import sqlite3
import warnings
from pathlib import Path
from typing import Any, Optional

from .connection import DatabaseConnection


class SchemaManager:
    """
    지식 그래프 및 벡터 저장을 위한 데이터베이스 스키마를 관리합니다.
    """

    CURRENT_SCHEMA_VERSION = 3  # 현재 스키마 버전

    def __init__(self, db_path: str | Path):
        """
        스키마 관리자를 초기화합니다.
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_connection = DatabaseConnection(db_path)

    def initialize_schema(self) -> None:
        """
        모든 필요한 테이블, 인덱스 및 트리거를 생성하여 데이터베이스 스키마를 초기화합니다.
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
            # 초기 스키마 버전 설정
            self._update_schema_version(conn, 1)

    def _create_schema_version_table(self, conn: sqlite3.Connection) -> None:
        """스키마 버전 추적 테이블을 생성합니다."""
        conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1), -- 하나의 행만 허용
            version INTEGER NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """
        )

    def _create_entity_tables(self, conn: sqlite3.Connection) -> None:
        """엔티티(노드) 관련 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 엔티티 (노드) 테이블
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            uuid TEXT UNIQUE NOT NULL, -- 외부 참조를 위한 안정적인 식별자
            name TEXT,
            type TEXT NOT NULL,
            properties JSON,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- 엔티티 테이블 인덱스
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_uuid ON entities(uuid);
        -- updated_at 타임스탬프 업데이트 트리거
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
        """이진 엣지 관련 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 이진 관계 (엣지) 테이블
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
        -- 엣지 테이블 인덱스
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_relation_type ON edges(relation_type);
        CREATE INDEX IF NOT EXISTS idx_edges_source_relation ON edges(source_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_edges_target_relation ON edges(target_id, relation_type);
        -- updated_at 타임스탬프 업데이트 트리거
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
        """하이퍼엣지(n-ary 관계) 관련 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 하이퍼엣지 테이블 (n-ary 관계용)
        CREATE TABLE IF NOT EXISTS hyperedges (
            id INTEGER PRIMARY KEY,
            hyperedge_type TEXT NOT NULL,
            properties JSON,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- 하이퍼엣지 멤버 테이블 (엔티티를 하이퍼엣지에 연결)
        CREATE TABLE IF NOT EXISTS hyperedge_members (
            hyperedge_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (hyperedge_id, entity_id, role),
            FOREIGN KEY (hyperedge_id) REFERENCES hyperedges(id) ON DELETE CASCADE,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- 하이퍼엣지 테이블 인덱스
        CREATE INDEX IF NOT EXISTS idx_hyperedges_type ON hyperedges(hyperedge_type);
        CREATE INDEX IF NOT EXISTS idx_hyperedge_members_entity ON hyperedge_members(entity_id);
        CREATE INDEX IF NOT EXISTS idx_hyperedge_members_role ON hyperedge_members(role);
        -- updated_at 타임스탬프 업데이트 트리거
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
        """문서 관련 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 문서 테이블
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
        -- 문서 테이블 인덱스
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
        -- updated_at 타임스탬프 업데이트 및 버전 증가 트리거
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
        """관찰 관련 테이블을 생성합니다 (콜드 데이터 저장용)."""
        conn.executescript(
            """
        -- 관찰 테이블 (콜드/히스토리 데이터용)
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata JSON,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- 관찰 테이블 인덱스
        CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
        CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at);
        """
        )

    def _create_embedding_tables(self, conn: sqlite3.Connection) -> None:
        """벡터 임베딩 관련 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 노드 임베딩 테이블
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL, -- 벡터의 이진 표현
            dimensions INTEGER NOT NULL, -- 벡터의 차원 수
            model_info TEXT NOT NULL, -- 임베딩 모델 정보
            embedding_version INTEGER NOT NULL DEFAULT 1, -- 버전 관리/추적용
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (node_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        -- 엣지 임베딩 테이블
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
        -- 하이퍼엣지 임베딩 테이블
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
        -- updated_at 타임스탬프 업데이트 트리거
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
        """아웃박스 패턴을 사용한 벡터-DB 동기화 테이블을 생성합니다."""
        conn.executescript(
            """
        -- 벡터 작업 아웃박스 테이블 (비동기 처리용)
        CREATE TABLE IF NOT EXISTS vector_outbox (
            id INTEGER PRIMARY KEY,
            operation_type TEXT NOT NULL, -- 'insert', 'update', 'delete'
            entity_type TEXT NOT NULL, -- 'node', 'edge', 'hyperedge'
            entity_id INTEGER NOT NULL,
            model_info TEXT,
            status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
            correlation_id TEXT, -- 관련 작업 추적용
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        -- 효율적인 아웃박스 처리를 위한 인덱스
        CREATE INDEX IF NOT EXISTS idx_vector_outbox_status ON vector_outbox(status);
        CREATE INDEX IF NOT EXISTS idx_vector_outbox_entity ON vector_outbox(entity_type, entity_id);
        -- 동기화 실패 로깅 테이블
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
        -- vector_outbox updated_at 업데이트 트리거
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
        스키마 버전을 업데이트하거나 삽입합니다.
        Args:
            conn: SQLite 연결
            version: 새 스키마 버전 번호
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
        현재 스키마 버전을 가져옵니다.
        Returns:
            현재 스키마 버전 번호 또는 설정되지 않은 경우 0
        """
        with self.db_connection as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT version FROM schema_version WHERE id = 1")
                result = cursor.fetchone()
                return result[0] if result else 0
            except sqlite3.OperationalError:
                # 스키마 버전 테이블이 존재하지 않으면 0을 반환
                return 0

    def migrate_schema(self, target_version: Optional[int] = None) -> bool:
        """
        스키마를 대상 버전으로 마이그레이션합니다.
        Args:
            target_version: 대상 스키마 버전 (기본값은 최신)
        Returns:
            마이그레이션 성공 시 True
        Raises:
            ValueError: 대상 버전이 잘못된 경우
            sqlite3.Error: 마이그레이션 실패 시
        """
        if target_version is None:
            target_version = self.CURRENT_SCHEMA_VERSION
        current_version = self.get_schema_version()
        if current_version == target_version:
            return True
        if target_version < current_version:
            raise ValueError(
                f"버전 {current_version}에서 {target_version}로의 다운그레이드는 지원되지 않습니다"
            )
        if target_version > self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"대상 버전 {target_version}이(가) 최신 버전 {self.CURRENT_SCHEMA_VERSION}보다 높습니다"
            )
        # 단계별로 마이그레이션 적용
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
                raise sqlite3.Error(
                    f"버전 {target_version}으로의 마이그레이션 실패: {exception}"
                ) from exception
        finally:
            conn.close()

    def _apply_migration(self, conn: sqlite3.Connection, version: int) -> None:
        """
        특정 마이그레이션 버전을 적용합니다.
        Args:
            conn: SQLite 연결
            version: 적용할 마이그레이션 버전
        """
        if version == 1:
            # 초기 스키마 생성
            self._create_entity_tables(conn)
            self._create_edge_tables(conn)
            self._create_hyperedge_tables(conn)
            self._create_embedding_tables(conn)
            self._create_observation_tables(conn)
        elif version == 2:
            # JSON 최적화를 위한 생성된 열 추가 (이미 추가되지 않은 경우)
            self._add_json_optimization_columns(conn)
        elif version == 3:
            # 문서 테이블 추가
            self._create_document_tables(conn)
        else:
            raise ValueError(f"알 수 없는 마이그레이션 버전: {version}")

    def _add_json_optimization_columns(self, conn: sqlite3.Connection) -> None:
        """존재하지 않는 경우 JSON 최적화 열을 추가합니다."""
        try:
            # 열이 이미 존재하는지 확인
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(entities)")
            columns = [row[1] for row in cursor.fetchall()]
            if "json_text_content" not in columns:
                conn.executescript(
                    """
                -- 엔티티에 대한 생성된 열 추가
                ALTER TABLE entities ADD COLUMN json_text_content TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.bio')) STORED;
                ALTER TABLE entities ADD COLUMN json_category TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.category')) STORED;
                ALTER TABLE entities ADD COLUMN json_status TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.status')) STORED;
                -- 인덱스 추가
                CREATE INDEX IF NOT EXISTS idx_entities_json_text ON entities(json_text_content) WHERE json_text_content IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_entities_json_category ON entities(json_category) WHERE json_category IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_entities_json_status ON entities(json_status) WHERE json_status IS NOT NULL;
                """
                )
            # 엣지 테이블 확인
            cursor.execute("PRAGMA table_info(edges)")
            columns = [row[1] for row in cursor.fetchall()]
            if "json_weight" not in columns:
                conn.executescript(
                    """
                -- 엣지에 대한 생성된 열 추가
                ALTER TABLE edges ADD COLUMN json_weight REAL
                    GENERATED ALWAYS AS (CAST(JSON_EXTRACT(properties, '$.weight') AS REAL)) STORED;
                ALTER TABLE edges ADD COLUMN json_since TEXT
                    GENERATED ALWAYS AS (JSON_EXTRACT(properties, '$.since')) STORED;
                ALTER TABLE edges ADD COLUMN json_confidence REAL
                    GENERATED ALWAYS AS (CAST(JSON_EXTRACT(properties, '$.confidence') AS REAL)) STORED;
                -- 인덱스 추가
                CREATE INDEX IF NOT EXISTS idx_edges_json_weight ON edges(json_weight) WHERE json_weight IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_edges_json_since ON edges(json_since) WHERE json_since IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_edges_json_confidence ON edges(json_confidence) WHERE json_confidence IS NOT NULL;
                """
                )
        except sqlite3.Error as exception:
            # 생성된 열이 지원되지 않는 경우 (이전 SQLite), 조용히 건너뜁니다
            warnings.warn(f"JSON 최적화 열을 추가할 수 없습니다: {exception}", stacklevel=2)

    def backup_schema(self, backup_path: str) -> bool:
        """
        현재 데이터베이스의 백업을 생성합니다.
        Args:
            backup_path: 백업 파일 경로
        Returns:
            백업 성공 시 True
        """
        try:
            with self.db_connection as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            return True
        except Exception as exception:
            raise sqlite3.Error(f"백업 실패: {exception}") from exception

    def validate_schema(self) -> dict:
        """
        현재 스키마 무결성을 검증합니다.
        Returns:
            검증 결과가 포함된 사전
        """
        # 연결 작업 전에 먼저 버전 가져오기
        try:
            version = self.get_schema_version()
        except (sqlite3.Error, Exception):
            version = 0
        results: dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "version": version}
        conn = self.db_connection.connect()
        try:
            # 외래 키 무결성 확인
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_key_check")
            fk_errors = cursor.fetchall()
            if fk_errors:
                results["valid"] = False
                error_list = [f"외래 키 오류: {error}" for error in fk_errors]
                results["errors"].extend(error_list)
            # 테이블 무결성 확인
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
                        results["errors"].append(f"{table}에 대한 무결성 검사 실패: {integrity}")
                except sqlite3.Error:
                    # 테이블이 존재하지 않을 수 있으므로 건너뜁니다
                    pass
            # 인덱스 확인
            cursor.execute("PRAGMA index_list(entities)")
            if not cursor.fetchall():
                warnings_list = results["warnings"]
                warnings_list.append("엔티티 테이블에서 인덱스를 찾을 수 없습니다")
        except sqlite3.Error as exception:
            results["valid"] = False
            errors_list = results["errors"]
            errors_list.append(f"스키마 검증 오류: {exception}")
        finally:
            conn.close()
        return results
