"""
벡터 처리 인프라 예외.

이 예외들은 벡터 연산, 임베딩 생성,
벡터 데이터베이스 오류를 처리합니다.
"""

from typing import Any, Optional

from ..exceptions.base import InfrastructureException
from ..exceptions.data import DataValidationException


class VectorException(InfrastructureException):
    """
    벡터 처리 오류의 기본 예외.

    벡터 연산, 인덱싱, 벡터 데이터 관리 관련
    문제를 다룹니다.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        vector_dimension: Optional[int] = None,
        error_code: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 예외를 초기화합니다.

        Args:
            operation: 수행 중인 벡터 연산
            message: 상세 오류 메시지
            vector_dimension: 관련된 경우 벡터 차원
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.operation = operation
        self.vector_dimension = vector_dimension

        full_message = f"벡터 {operation} 실패: {message}"
        if vector_dimension:
            full_message += f" (차원: {vector_dimension})"

        super().__init__(
            message=full_message,
            error_code=error_code or "VECTOR_ERROR",
            context=context,
            original_error=original_error,
        )


class EmbeddingGenerationException(VectorException):
    """
    임베딩 생성 실패.

    텍스트-벡터 변환, 모델 로딩, 임베딩 계산 중
    오류를 처리합니다.
    """

    def __init__(
        self,
        text: str,
        model_name: str,
        message: str,
        expected_dimension: Optional[int] = None,
        actual_dimension: Optional[int] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        임베딩 생성 예외를 초기화합니다.

        Args:
            text: 입력 텍스트 (로깅을 위해 잘림)
            model_name: 사용된 임베딩 모델
            message: 상세 오류 메시지
            expected_dimension: 예상 벡터 차원
            actual_dimension: 실제 수신된 벡터 차원
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.text = text[:200] + "..." if len(text) > 200 else text
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension

        full_message = f"모델 '{model_name}'에 대한 임베딩 생성 실패: {message}"
        if expected_dimension and actual_dimension:
            full_message += f" (예상 차원: {expected_dimension}, 실제: {actual_dimension})"

        super().__init__(
            operation="embedding generation",
            message=full_message,
            vector_dimension=expected_dimension,
            error_code="EMBEDDING_GENERATION_FAILED",
            context=context,
            original_error=original_error,
        )


class VectorDimensionException(DataValidationException):
    """
    벡터 차원 불일치.

    벡터 차원이 예상 값과 일치하지 않거나 호환되지 않는
    경우를 처리합니다.
    """

    def __init__(
        self,
        expected_dimension: int,
        actual_dimension: int,
        operation: str,
        vector_id: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 차원 예외를 초기화합니다.

        Args:
            expected_dimension: 예상 벡터 차원
            actual_dimension: 실제 벡터 차원
            operation: 수행 중인 작업
            vector_id: 사용 가능한 경우 벡터 식별자
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.expected_dimension = expected_dimension
        self.actual_dimension = actual_dimension
        self.operation = operation
        self.vector_id = vector_id

        message = (
            f"{operation}에서 벡터 차원 불일치: 예상 {expected_dimension}, 실제 {actual_dimension}"
        )
        if vector_id:
            message += f" (벡터: {vector_id})"

        super().__init__(
            field="vector_dimension",
            value=actual_dimension,
            expected_format=f"{expected_dimension}D 벡터",
            message=message,
            error_code="VECTOR_DIMENSION_MISMATCH",
            context=context,
            original_error=original_error,
        )


class VectorSearchException(VectorException):
    """
    벡터 검색 및 유사도 오류.

    벡터 검색, 유사도 계산, 인덱스 쿼리 중
    실패를 처리합니다.
    """

    def __init__(
        self,
        query_vector_dimension: int,
        index_dimension: Optional[int] = None,
        search_params: dict[str, Any] | None = None,
        message: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 검색 예외를 초기화합니다.

        Args:
            query_vector_dimension: 쿼리 벡터 차원
            index_dimension: 인덱스 벡터 차원
            search_params: 사용된 검색 매개변수
            message: 선택적 사용자 지정 메시지
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.query_vector_dimension = query_vector_dimension
        self.index_dimension = index_dimension
        self.search_params = search_params or {}

        if message is None:
            message = "벡터 검색 실패"
            if index_dimension and query_vector_dimension != index_dimension:
                message += (
                    f": 차원 불일치 (쿼리: {query_vector_dimension}, 인덱스: {index_dimension})"
                )

        super().__init__(
            operation="vector search",
            message=message,
            vector_dimension=query_vector_dimension,
            error_code="VECTOR_SEARCH_FAILED",
            context=context,
            original_error=original_error,
        )


class VectorIndexException(VectorException):
    """
    벡터 인덱스 오류.

    벡터 인덱스 생성, 업데이트, 유지보수 작업 관련
    문제를 처리합니다.
    """

    def __init__(
        self,
        index_name: str,
        operation: str,
        message: str,
        vector_count: Optional[int] = None,
        dimension: Optional[int] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 인덱스 예외를 초기화합니다.

        Args:
            index_name: 벡터 인덱스의 이름
            operation: 수행 중인 인덱스 작업
            message: 상세 오류 메시지
            vector_count: 인덱스의 벡터 수
            dimension: 벡터 차원
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.index_name = index_name
        self.vector_count = vector_count

        full_message = f"벡터 인덱스 '{index_name}' {operation} 실패: {message}"
        if vector_count:
            full_message += f" (벡터: {vector_count})"

        super().__init__(
            operation=f"index {operation}",
            message=full_message,
            vector_dimension=dimension,
            error_code="VECTOR_INDEX_ERROR",
            context=context,
            original_error=original_error,
        )


class VectorStorageException(VectorException):
    """
    벡터 저장 및 영속성 오류.

    저장 시스템에서 벡터 데이터를 저장, 검색, 관리하는 데
    실패하는 경우를 처리합니다.
    """

    def __init__(
        self,
        storage_type: str,
        operation: str,
        entity_id: Optional[str] = None,
        message: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 저장 예외를 초기화합니다.

        Args:
            storage_type: 저장소 유형 (SQLite, 파일 등)
            operation: 저장소 작업
            entity_id: 관련된 경우 엔티티 식별자
            message: 선택적 사용자 지정 메시지
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.storage_type = storage_type
        self.entity_id = entity_id

        if message is None:
            message = f"벡터 저장 작업 '{operation}'이(가) {storage_type}에서 실패했습니다."
            if entity_id:
                message += f" (엔티티: {entity_id})"

        super().__init__(
            operation=f"storage {operation}",
            message=message,
            error_code="VECTOR_STORAGE_ERROR",
            context=context,
            original_error=original_error,
        )


class VectorNormalizationException(VectorException):
    """
    벡터 정규화 오류.

    벡터 정규화, 유효성 검사, 전처리 작업 관련
    문제를 처리합니다.
    """

    def __init__(
        self,
        vector_shape: tuple,
        normalization_type: str,
        message: str,
        vector_stats: dict[str, float] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        벡터 정규화 예외를 초기화합니다.

        Args:
            vector_shape: 문제가 있는 벡터의 모양
            normalization_type: 시도된 정규화 유형
            message: 상세 오류 메시지
            vector_stats: 벡터 통계 (평균, 표준편차 등)
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.vector_shape = vector_shape
        self.normalization_type = normalization_type
        self.vector_stats = vector_stats or {}

        full_message = f"벡터 정규화({normalization_type})가 모양 {vector_shape}에 대해 실패했습니다: {message}"

        super().__init__(
            operation=f"normalization ({normalization_type})",
            message=full_message,
            vector_dimension=vector_shape[0] if vector_shape else None,
            error_code="VECTOR_NORMALIZATION_FAILED",
            context=context,
            original_error=original_error,
        )
