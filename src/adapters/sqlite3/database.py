"""
Database 포트의 SQLite 구현.
"""

import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from src.common.config.database import DatabaseConfig
from src.ports.database import Database, DatabaseMaintenance

from .connection import DatabaseConnection


class SQLiteDatabase(Database, DatabaseMaintenance):
    """
    Database 포트의 SQLite 구현.
    이 어댑터는 SQLite를 기본 스토리지 엔진으로 사용하여 데이터베이스 작업의
    구체적인 구현을 제공합니다.
    """

    def __init__(
        self,
        config: DatabaseConfig | None = None,
        db_path: str | None = None,
        optimize: bool | None = None,
    ):
        """
        SQLite 데이터베이스 어댑터를 초기화합니다.
        Args:
            config: 데이터베이스 설정 객체
            db_path: SQLite 데이터베이스 파일 경로 (사용 중단됨, config 사용 권장)
            optimize: 최적화 PRAGMA 적용 여부 (사용 중단됨, config 사용 권장)
        """
        if config is None:
            config = DatabaseConfig()
        self.db_path = Path(db_path or config.db_path)
        self.optimize = optimize if optimize is not None else config.optimize
        self.timeout = config.timeout
        self.check_same_thread = config.check_same_thread
        self.max_connections = config.max_connections
        self._connection_manager = DatabaseConnection(str(self.db_path), self.optimize)
        self._connection: Connection | None = None
        self._active_transactions: dict[str, sqlite3.Connection] = {}

    # 연결 관리
    async def connect(self) -> bool:
        """
        데이터베이스 연결을 설정합니다.
        Returns:
            연결 성공 시 True
        """
        try:
            self._connection = self._connection_manager.connect()
            return True
        except Exception:
            return False

    async def disconnect(self) -> bool:
        """
        데이터베이스 연결을 닫습니다.
        Returns:
            연결 해제 성공 시 True
        """
        try:
            for transaction_conn in self._active_transactions.values():
                try:
                    transaction_conn.rollback()
                    transaction_conn.close()
                except Exception:
                    pass
            self._active_transactions.clear()
            self._connection_manager.close()
            self._connection = None
            return True
        except Exception:
            return False

    async def is_connected(self) -> bool:
        """
        데이터베이스에 연결되었는지 확인합니다.
        Returns:
            데이터베이스에 연결된 경우 True
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
        데이터베이스에 핑을 보내 연결 상태를 확인합니다.
        Returns:
            데이터베이스가 응답하면 True
        """
        return await self.is_connected()

    # 트랜잭션 관리
    @asynccontextmanager
    async def transaction(self):  # pylint: disable=invalid-overridden-method
        """
        데이터베이스 트랜잭션 컨텍스트를 생성합니다.
        Yields:
            트랜잭션 컨텍스트
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
        새 트랜잭션을 시작합니다.
        Returns:
            트랜잭션 ID
        """
        if not self._connection:
            raise RuntimeError("데이터베이스가 연결되지 않았습니다")
        transaction_id = str(uuid.uuid4())
        self._active_transactions[transaction_id] = self._connection
        self._connection.execute("BEGIN")
        return transaction_id

    async def commit_transaction(self, transaction_id: str) -> bool:
        """
        트랜잭션을 커밋합니다.
        Args:
            transaction_id: 커밋할 트랜잭션 ID
        Returns:
            커밋 성공 시 True
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
        트랜잭션을 롤백합니다.
        Args:
            transaction_id: 롤백할 트랜잭션 ID
        Returns:
            롤백 성공 시 True
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

    # 쿼리 실행
    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        SELECT 쿼리를 실행합니다.
        Args:
            query: 실행할 SQL 쿼리
            parameters: 선택적 쿼리 매개변수
            transaction_id: 선택적 트랜잭션 ID
        Returns:
            딕셔너리 리스트 형태의 쿼리 결과
        """
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("데이터베이스가 연결되지 않았습니다")
        cursor = connection.cursor()
        try:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            columns = (
                [description[0] for description in cursor.description] if cursor.description else []
            )
            rows = cursor.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]
        finally:
            cursor.close()

    async def execute_command(
        self,
        command: str,
        parameters: dict[str, Any] | None = None,
        transaction_id: str | None = None,
    ) -> int:
        """
        SELECT가 아닌 명령(INSERT, UPDATE, DELETE)을 실행합니다.
        Args:
            command: 실행할 SQL 명령
            parameters: 선택적 명령 매개변수
            transaction_id: 선택적 트랜잭션 ID
        Returns:
            영향을 받은 행의 수
        """
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("데이터베이스가 연결되지 않았습니다")
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
        commands: list[str],
        parameters: list[dict[str, Any]] | None = None,
        transaction_id: str | None = None,
    ) -> list[int]:
        """
        여러 명령을 일괄 실행합니다.
        Args:
            commands: SQL 명령 목록
            parameters: 각 명령에 대한 선택적 매개변수 목록
            transaction_id: 선택적 트랜잭션 ID
        Returns:
            영향을 받은 행 수의 목록
        """
        connection = self._get_connection(transaction_id)
        if not connection:
            raise RuntimeError("데이터베이스가 연결되지 않았습니다")
        results = []
        for i, command in enumerate(commands):
            cursor = connection.cursor()
            try:
                cmd_params = parameters[i] if parameters and i < len(parameters) else None
                if cmd_params:
                    cursor.execute(command, cmd_params)
                else:
                    cursor.execute(command)
                results.append(cursor.rowcount)
            finally:
                cursor.close()
        return results

    # 스키마 관리
    async def create_table(
        self, table_name: str, schema: dict[str, Any], if_not_exists: bool = True
    ) -> bool:
        """
        데이터베이스 테이블을 생성합니다.
        Args:
            table_name: 테이블 이름
            schema: 테이블 스키마 정의
            if_not_exists: IF NOT EXISTS 절 사용 여부
        Returns:
            테이블 생성 성공 시 True
        """
        try:
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
        데이터베이스 테이블을 삭제합니다.
        Args:
            table_name: 테이블 이름
            if_exists: IF EXISTS 절 사용 여부
        Returns:
            테이블 삭제 성공 시 True
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
        테이블이 존재하는지 확인합니다.
        Args:
            table_name: 테이블 이름
        Returns:
            테이블이 존재하면 True
        """
        try:
            result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                {"name": table_name},
            )
            return len(result) > 0
        except Exception:
            return False

    async def get_table_schema(self, table_name: str) -> dict[str, Any] | None:
        """
        테이블의 스키마를 가져옵니다.
        Args:
            table_name: 테이블 이름
        Returns:
            테이블 스키마 또는 테이블이 없으면 None
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
        columns: list[str],
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> bool:
        """
        데이터베이스 인덱스를 생성합니다.
        Args:
            index_name: 인덱스 이름
            table_name: 인덱싱할 테이블
            columns: 인덱스에 포함할 열
            unique: 인덱스를 고유하게 만들지 여부
            if_not_exists: IF NOT EXISTS 절 사용 여부
        Returns:
            인덱스 생성 성공 시 True
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
        데이터베이스 인덱스를 삭제합니다.
        Args:
            index_name: 인덱스 이름
            if_exists: IF EXISTS 절 사용 여부
        Returns:
            인덱스 삭제 성공 시 True
        """
        try:
            if_exists_clause = "IF EXISTS " if if_exists else ""
            sql = f"DROP INDEX {if_exists_clause}{index_name}"
            await self.execute_command(sql)
            return True
        except Exception:
            return False

    # 데이터베이스 유지보수
    async def vacuum(self) -> bool:
        """
        데이터베이스 vacuum 작업을 수행합니다.
        Returns:
            vacuum 성공 시 True
        """
        try:
            await self.execute_command("VACUUM")
            return True
        except Exception:
            return False

    async def analyze(self, table_name: str | None = None) -> bool:
        """
        데이터베이스 통계를 분석합니다.
        Args:
            table_name: 분석할 특정 테이블 (선택 사항)
        Returns:
            분석 성공 시 True
        """
        try:
            if table_name:
                await self.execute_command(f"ANALYZE {table_name}")
            else:
                await self.execute_command("ANALYZE")
            return True
        except Exception:
            return False

    async def get_database_info(self) -> dict[str, Any]:
        """
        데이터베이스 정보 및 통계를 가져옵니다.
        Returns:
            데이터베이스 정보
        """
        try:
            info = {
                "path": str(self.db_path),
                "size_bytes": (self.db_path.stat().st_size if self.db_path.exists() else 0),
            }
            # SQLite 버전 및 설정 가져오기
            version_result = await self.execute_query("SELECT sqlite_version()")
            if version_result:
                info["sqlite_version"] = version_result[0]["sqlite_version()"]
            # pragma 설정 가져오기
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
            return {"error": "데이터베이스 정보를 가져오는 데 실패했습니다"}

    async def get_table_info(self, table_name: str) -> dict[str, Any] | None:
        """
        특정 테이블에 대한 정보를 가져옵니다.
        Args:
            table_name: 테이블 이름
        Returns:
            테이블 정보 또는 테이블이 없으면 None
        """
        try:
            # 테이블 존재 여부 확인
            if not await self.table_exists(table_name):
                return None
            info: dict[str, Any] = {"name": table_name}
            # 행 수 가져오기
            count_result = await self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            if count_result:
                info["row_count"] = count_result[0]["count"]
            # 스키마 가져오기
            schema = await self.get_table_schema(table_name)
            if schema:
                info["schema"] = schema
            return info
        except Exception:
            return None

    # 상태 및 진단
    async def health_check(self) -> dict[str, Any]:
        """
        데이터베이스 상태 확인을 수행합니다.
        Returns:
            상태 정보
        """
        health: dict[str, Any] = {
            "connected": await self.is_connected(),
            "file_exists": self.db_path.exists(),
            "readable": False,
            "writable": False,
        }
        if health["connected"]:
            try:
                # 읽기 테스트
                await self.execute_query("SELECT 1")
                health["readable"] = True
                # 쓰기 테스트 (임시 테이블 생성 및 삭제)
                await self.execute_command("CREATE TEMP TABLE health_check_temp (id INTEGER)")
                await self.execute_command("DROP TABLE health_check_temp")
                health["writable"] = True
            except Exception as exception:
                health["error"] = str(exception)
        health["status"] = (
            "healthy"
            if all([health["connected"], health["readable"], health["writable"]])
            else "unhealthy"
        )
        return health

    async def get_connection_info(self) -> dict[str, Any]:
        """
        연결 정보 및 상태를 가져옵니다.
        Returns:
            연결 정보
        """
        return {
            "connected": await self.is_connected(),
            "db_path": str(self.db_path),
            "optimize": self.optimize,
            "active_transactions": len(self._active_transactions),
            "transaction_ids": list(self._active_transactions.keys()),
        }

    async def get_performance_stats(self) -> dict[str, Any]:
        """
        데이터베이스 성능 통계를 가져옵니다.
        Returns:
            성능 통계
        """
        try:
            stats: dict[str, Any] = {}
            # 데이터베이스 크기
            if self.db_path.exists():
                stats["file_size_bytes"] = self.db_path.stat().st_size
            # 테이블 통계
            tables_result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            stats["table_count"] = len(tables_result)
            tables_dict: dict[str, Any] = {}
            for table_row in tables_result:
                table_name = table_row["name"]
                table_info = await self.get_table_info(table_name)
                if table_info:
                    tables_dict[table_name] = {"row_count": table_info.get("row_count", 0)}
            stats["tables"] = tables_dict
            return stats
        except Exception:
            return {"error": "성능 통계를 가져오는 데 실패했습니다"}

    @property
    def connection(self) -> Connection | None:
        """
        SQLite 연결을 가져옵니다.
        Returns:
            SQLite 연결 또는 연결되지 않은 경우 None
        """
        return self._connection

    def _get_connection(self, transaction_id: str | None = None) -> Connection | None:
        """
        트랜잭션에 적합한 연결을 가져옵니다.
        Args:
            transaction_id: 선택적 트랜잭션 ID
        Returns:
            SQLite 연결 또는 None
        """
        if transaction_id and transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id]
        return self._connection
