"""
검색 관련 설정들을 정의합니다.
"""

from dataclasses import dataclass

# 기본 유사도 임계값
DEFAULT_SIMILARITY_THRESHOLD = 0.5


@dataclass
class SearchConfig:
    """검색 설정을 관리하는 클래스"""

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_results: int = 10
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    mmr_lambda: float = 0.6

    def __post_init__(self):
        """설정 검증"""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 0.0과 1.0 사이여야 합니다")
        if self.max_results <= 0:
            raise ValueError("max_results는 양수여야 합니다")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("keyword_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda는 0.0과 1.0 사이여야 합니다")


# 향후 다른 검색 관련 설정들이 추가될 수 있습니다.
