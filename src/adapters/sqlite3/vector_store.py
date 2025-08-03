"""
sqlite-vec 확장을 사용한 VectorStore 포트의 SQLite 구현.
"""

from src.ports.vector_store import VectorStore

from .vector_reader_impl import SQLiteVectorReader
from .vector_retriever_impl import SQLiteVectorRetriever
from .vector_writer_impl import SQLiteVectorWriter


class SQLiteVectorStore(SQLiteVectorWriter, SQLiteVectorReader, SQLiteVectorRetriever, VectorStore):
    """
    VectorStore 포트의 SQLite 구현.

    이 클래스는 다중 상속을 통해 분리된 구현체들을 통합하여
    완전한 VectorStore 인터페이스를 제공합니다.

    - SQLiteVectorWriter: 데이터 추가/수정/삭제
    - SQLiteVectorReader: 데이터 조회/검색
    - SQLiteVectorRetriever: 고급 검색/리트리벌
    """

    def __init__(self, db_path: str, table_name: str = "vectors", optimize: bool = True):
        """
        SQLite 벡터 저장소 어댑터를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            table_name: 벡터를 저장할 테이블 이름
            optimize: 최적화 PRAGMA 적용 여부
        """
        # 모든 베이스 클래스들이 동일한 초기화를 사용하므로 한 번만 호출
        SQLiteVectorWriter.__init__(self, db_path, table_name, optimize)
