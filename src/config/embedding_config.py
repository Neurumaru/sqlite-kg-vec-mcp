"""
임베딩 및 배치 처리 관련 설정들을 정의합니다.
"""

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """임베딩 및 배치 처리 설정을 관리하는 클래스"""

    # 배치 크기 설정
    default_batch_size: int = 100
    knowledge_extraction_batch_size: int = 10
    embedding_batch_size: int = 1000
    hnsw_index_batch_size: int = 1000
    outbox_process_batch_size: int = 100

    # 임베딩 관련 설정
    max_text_length: int = 8192  # 임베딩할 텍스트 최대 길이
    min_text_length: int = 1  # 임베딩할 텍스트 최소 길이

    def __post_init__(self):
        """설정 검증"""
        # 배치 크기 검증
        if self.default_batch_size <= 0:
            raise ValueError("default_batch_size는 양수여야 합니다")
        if self.knowledge_extraction_batch_size <= 0:
            raise ValueError("knowledge_extraction_batch_size는 양수여야 합니다")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size는 양수여야 합니다")
        if self.hnsw_index_batch_size <= 0:
            raise ValueError("hnsw_index_batch_size는 양수여야 합니다")
        if self.outbox_process_batch_size <= 0:
            raise ValueError("outbox_process_batch_size는 양수여야 합니다")

        # 텍스트 길이 검증
        if self.min_text_length < 0:
            raise ValueError("min_text_length는 0 이상이어야 합니다")
        if self.max_text_length <= self.min_text_length:
            raise ValueError("max_text_length는 min_text_length보다 커야 합니다")


# 기본 설정 인스턴스
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
