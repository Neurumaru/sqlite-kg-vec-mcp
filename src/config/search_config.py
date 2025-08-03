"""
검색 관련 설정들을 정의합니다.
"""

from dataclasses import dataclass

# 기본 유사도 임계값
DEFAULT_SIMILARITY_THRESHOLD = 0.5


@dataclass
class SearchConfig:
    """검색 설정을 관리하는 클래스"""

    # 기본 검색 설정
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_results: int = 10

    # 하이브리드 검색 가중치
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # 문서 검색 가중치 (제목 vs 내용)
    document_title_weight: float = 0.7
    document_content_weight: float = 0.3

    # 노드 검색 가중치 (이름 vs 설명 vs 임베딩)
    node_name_weight: float = 0.6
    node_description_weight: float = 0.3
    node_embedding_weight: float = 0.1

    # MMR(Maximal Marginal Relevance) 설정
    mmr_lambda: float = 0.5  # 관련성 vs 다양성 균형 (0.5가 기본값)

    # 벡터 검색 임계값 설정
    score_threshold: float = 0.5

    # 노드 유사도 임계값
    node_similarity_threshold: float = 0.7

    def __post_init__(self):
        """설정 검증"""
        # 기본 설정 검증
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 0.0과 1.0 사이여야 합니다")
        if self.max_results <= 0:
            raise ValueError("max_results는 양수여야 합니다")

        # 하이브리드 검색 가중치 검증
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("keyword_weight는 0.0과 1.0 사이여야 합니다")

        # 문서 검색 가중치 검증
        if not 0.0 <= self.document_title_weight <= 1.0:
            raise ValueError("document_title_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.document_content_weight <= 1.0:
            raise ValueError("document_content_weight는 0.0과 1.0 사이여야 합니다")
        if abs(self.document_title_weight + self.document_content_weight - 1.0) > 1e-6:
            raise ValueError("document_title_weight + document_content_weight는 1.0이어야 합니다")

        # 노드 검색 가중치 검증
        if not 0.0 <= self.node_name_weight <= 1.0:
            raise ValueError("node_name_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.node_description_weight <= 1.0:
            raise ValueError("node_description_weight는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.node_embedding_weight <= 1.0:
            raise ValueError("node_embedding_weight는 0.0과 1.0 사이여야 합니다")
        total_node_weight = (
            self.node_name_weight + self.node_description_weight + self.node_embedding_weight
        )
        if abs(total_node_weight - 1.0) > 1e-6:
            raise ValueError("노드 가중치들의 합은 1.0이어야 합니다")

        # MMR 및 임계값 검증
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold는 0.0과 1.0 사이여야 합니다")
        if not 0.0 <= self.node_similarity_threshold <= 1.0:
            raise ValueError("node_similarity_threshold는 0.0과 1.0 사이여야 합니다")


# 향후 다른 검색 관련 설정들이 추가될 수 있습니다.
