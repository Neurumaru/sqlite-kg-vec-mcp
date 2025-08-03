"""
벡터 저장소를 위한 통합 포트 인터페이스.
"""

from abc import ABC

from .vector_reader import VectorReader
from .vector_retriever import VectorRetriever
from .vector_writer import VectorWriter


class VectorStore(VectorWriter, VectorReader, VectorRetriever, ABC):
    """
    벡터 저장소의 통합 포트 인터페이스.

    헥사고날 아키텍처 원칙을 준수하며 외부 라이브러리 의존성이 없습니다.
    인터페이스 분리 원칙(ISP)에 따라 세 가지 역할을 조합합니다:
    - VectorWriter: 데이터 추가/수정/삭제
    - VectorReader: 데이터 조회/검색
    - VectorRetriever: 고급 검색/리트리벌

    구현체는 필요에 따라 개별 인터페이스만 구현할 수도 있습니다.
    """
