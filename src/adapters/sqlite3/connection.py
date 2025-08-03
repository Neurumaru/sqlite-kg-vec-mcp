"""
SQLite 데이터베이스 연결 관리 모듈.
"""

import datetime
import sqlite3
import warnings
from pathlib import Path
from typing import Optional

from src.common.observability import get_observable_logger
from src.domain.config.timeout_config import TimeoutConfig

from .exceptions import SQLiteConnectionException


# Python 3.12 호환성을 위한 사용자 지정 타임스탬프 변환 함수 정의
def adapt_datetime(dt: datetime.datetime) -> str:
    """SQLite 저장을 위해 datetime을 문자열로 변환합니다."""
    return dt.isoformat()


def convert_datetime(s: str | bytes) -> datetime.datetime | str | bytes:
    """SQLite의 문자열을 다시 datetime으로 변환합니다."""
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return datetime.datetime.fromisoformat(s)
    except (ValueError, AttributeError, UnicodeDecodeError) as exception:
        warnings.warn(f"datetime {s!r} 변환 실패: {exception}", stacklevel=2)
        return s


# 사용자 지정 타임스탬프 핸들러 등록
sqlite3.register_adapter(datetime.datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)


class DatabaseConnection:
    """
    최적화된 설정으로 SQLite 데이터베이스 연결을 관리합니다.
    """

    def __init__(
        self,
        db_path: str | Path,
        optimize: bool = True,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        """
        데이터베이스 연결을 초기화합니다.
        Args:
            db_path: SQLite 데이터베이스 파일 경로
            optimize: 최적화 PRAGMA 적용 여부
            timeout_config: 타임아웃 설정 객체
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self.optimize = optimize
        self.timeout_config = timeout_config or TimeoutConfig.from_env()
        self.logger = get_observable_logger("database_connection", "adapter")

    def connect(self) -> sqlite3.Connection:
        """
        SQLite 데이터베이스에 연결합니다.
        Returns:
            SQLite 연결 객체
        Raises:
            sqlite3.Error: 데이터베이스 연결 실패 시
            PermissionError: 데이터베이스 파일을 생성/접근할 수 없는 경우
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exception:
            raise PermissionError(
                f"데이터베이스 디렉토리 {self.db_path.parent}를 생성할 수 없습니다: {exception}"
            ) from exception
        try:
            # 데이터베이스에 연결
            self.connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                isolation_level=None,  # 트랜잭션을 명시적으로 관리할 것입니다.
                check_same_thread=False,  # 여러 스레드에서 사용 허용
                timeout=self.timeout_config.database_connection_timeout,
            )
            # 행을 딕셔너리로 반환하도록 설정
            self.connection.row_factory = sqlite3.Row
            # 연결 테스트
            self.connection.execute("SELECT 1").fetchone()
            # 요청된 경우 최적화 PRAGMA 적용
            if self.optimize:
                self._apply_optimizations()
            return self.connection
        except sqlite3.Error as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"SQLite 데이터베이스 연결 실패: {exception}",
                original_error=exception,
            ) from exception
        except PermissionError as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"데이터베이스 접근 권한 거부: {exception}",
                original_error=exception,
            ) from exception
        except Exception as exception:
            raise SQLiteConnectionException(
                db_path=str(self.db_path),
                message=f"연결 중 예기치 않은 오류 발생: {exception}",
                original_error=exception,
            ) from exception

    def _apply_optimizations(self) -> None:
        """PRAGMA 문을 통해 SQLite 최적화를 적용합니다."""
        if not self.connection:
            return
        cursor = self.connection.cursor()
        # 더 나은 동시성과 성능을 위한 WAL 모드
        cursor.execute("PRAGMA journal_mode=WAL;")
        # 즉각적인 잠금 오류를 방지하기 위한 busy_timeout 설정
        cursor.execute(f"PRAGMA busy_timeout={self.timeout_config.get_database_busy_timeout_ms()};")
        # 일반 동기화 모드 (내구성과 성능 간의 균형)
        cursor.execute("PRAGMA synchronous=NORMAL;")
        # 외래 키 제약 조건 활성화
        cursor.execute("PRAGMA foreign_keys=ON;")
        # 임시 저장을 위해 메모리 사용
        cursor.execute("PRAGMA temp_store=MEMORY;")
        # 더 나은 성능을 위한 더 큰 캐시 (32MB)
        cursor.execute("PRAGMA cache_size=-32000;")  # 음수는 킬로바이트를 의미
        cursor.close()

    def close(self) -> None:
        """열려 있는 경우 데이터베이스 연결을 닫습니다."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> sqlite3.Connection:
        """컨텍스트 관리자 진입 메서드."""
        if not self.connection:
            return self.connect()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 관리자 종료 메서드."""
        self.close()
