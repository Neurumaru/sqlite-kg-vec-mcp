"""
sqlite-vec 확장을 사용한 VectorStore 포트의 SQLite 구현.
"""

from src.config.search_config import SearchConfig
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

    def __init__(
        self,
        db_path: str,
        table_name: str = "vectors",
        optimize: bool = True,
        search_config: Optional[SearchConfig] = None,
    ):
        """
        SQLite 벡터 저장소 어댑터를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            table_name: 벡터를 저장할 테이블 이름
            optimize: 최적화 PRAGMA 적용 여부
            search_config: 검색 설정 (None인 경우 기본값 사용)
        """
        # SQLiteVectorWriter는 search_config가 필요 없으므로 기본 초기화
        SQLiteVectorWriter.__init__(self, db_path, table_name, optimize)
        # SQLiteVectorRetriever는 search_config가 필요하므로 별도 초기화
        SQLiteVectorRetriever.__init__(self, db_path, table_name, optimize, search_config)
