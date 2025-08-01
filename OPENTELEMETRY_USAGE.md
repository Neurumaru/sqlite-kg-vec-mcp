# OpenTelemetry 통합 사용 가이드

이 문서는 sqlite-kg-vec-mcp 프로젝트에서 OpenTelemetry를 사용하는 방법을 설명합니다.

## 1. 초기 설정

### 의존성 설치
```bash
# OpenTelemetry 패키지가 이미 pyproject.toml에 포함되어 있습니다
uv sync
```

### 환경변수 설정 (선택사항)
```bash
# .env 파일 또는 환경변수로 설정
export OTEL_SERVICE_NAME="sqlite-kg-vec-mcp"
export OTEL_SERVICE_VERSION="0.2.0"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # OTLP 서버 사용시
export OTEL_EXPORTER_OTLP_INSECURE="true"
```

## 2. 기본 초기화

### 애플리케이션 시작 시 초기화
```python
from src.common.observability.otel_setup import initialize_opentelemetry

# 애플리케이션 시작 시 한 번만 호출
def main():
    # 기본 초기화 (환경변수 사용)
    initialize_opentelemetry()
    
    # 또는 명시적 설정
    initialize_opentelemetry(
        service_name="my-service",
        service_version="1.0.0",
        endpoint="http://localhost:4317",  # OTLP 서버 주소
        enable_console=True,  # 개발 시 콘솔 출력
    )
```

## 3. 데코레이터 사용법

### 기본 트레이싱
```python
from src.common.observability.otel_decorators import traced

@traced("document_processing")
def process_document(document):
    # 함수 실행 시 자동으로 스팬 생성
    return processed_document

@traced()  # 함수명을 스팬 이름으로 사용
def analyze_text(text):
    return analysis_result
```

### 메트릭 수집
```python
from src.common.observability.otel_decorators import measured

@measured("api_calls", track_duration=True, track_calls=True)
def api_call(endpoint):
    # 호출 횟수와 실행 시간 자동 수집
    return response

@measured()  # 기본 설정 사용
def database_query(query):
    return results
```

### 통합 관찰 (트레이싱 + 메트릭)
```python
from src.common.observability.otel_decorators import observed

@observed(
    operation_name="knowledge_extraction",
    span_attributes={"model": "gpt-4", "type": "completion"}
)
def extract_knowledge(text):
    # 트레이싱과 메트릭을 모두 수집
    return knowledge_graph
```

## 4. 도메인별 데코레이터

### 데이터베이스 작업
```python
from src.common.observability.otel_decorators import trace_database_operation

@trace_database_operation("documents")
def save_document(document):
    # 데이터베이스 관련 속성이 자동으로 추가됨
    return document_id

@trace_database_operation("relationships")
def create_relationship(source, target):
    return relationship_id
```

### LLM 작업
```python
from src.common.observability.otel_decorators import trace_llm_operation

@trace_llm_operation("ollama-llama3")
def generate_response(prompt):
    # LLM 관련 속성이 자동으로 추가됨
    return response

@trace_llm_operation()  # 모델명 자동 감지 시도
def extract_entities(text):
    return entities
```

### 벡터 검색 작업
```python
from src.common.observability.otel_decorators import trace_vector_operation

@trace_vector_operation("similarity_search")
def find_similar_documents(query_vector):
    # 벡터 검색 관련 속성이 자동으로 추가됨
    return similar_docs

@trace_vector_operation("embedding")
def create_embedding(text):
    return vector
```

## 5. 수동 트레이싱 (고급)

### Context Manager 사용
```python
from src.common.observability.otel_setup import get_tracer, get_meter

def complex_operation():
    tracer = get_tracer(__name__)
    meter = get_meter(__name__)
    
    # 메트릭 인스트루먼트 생성
    counter = meter.create_counter("operations_total")
    
    with tracer.start_as_current_span("complex_task") as span:
        span.set_attribute("task.type", "complex")
        span.set_attribute("user.id", "user123")
        
        # 중첩된 스팬
        with tracer.start_as_current_span("subtask") as sub_span:
            sub_span.set_attribute("subtask.id", "sub1")
            # 작업 수행
            result = perform_subtask()
            
        # 이벤트 기록
        span.add_event("중간 체크포인트", {"checkpoint": "halfway"})
        
        # 메트릭 기록
        counter.add(1, {"operation": "complex"})
        
        return result
```

### 예외 처리
```python
def risky_operation():
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("risky_task") as span:
        try:
            result = perform_risky_task()
            span.set_status(trace.Status(trace.StatusCode.OK))
            return result
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

## 6. 실제 사용 예제

### DocumentProcessor에 적용
```python
from src.common.observability.otel_decorators import observed, trace_llm_operation

class DocumentProcessor:
    
    @observed("document_processing")
    def process_document(self, document):
        # 전체 문서 처리 과정 추적
        entities = self._extract_entities(document)
        relationships = self._extract_relationships(document)
        return self._create_knowledge_graph(entities, relationships)
    
    @trace_llm_operation("ollama-llama3")
    def _extract_entities(self, document):
        # LLM을 사용한 엔티티 추출 추적
        return entities
    
    @trace_database_operation("entities")
    def _save_entities(self, entities):
        # 데이터베이스 저장 작업 추적
        return saved_entities
```

### KnowledgeSearch에 적용
```python
from src.common.observability.otel_decorators import trace_vector_operation

class KnowledgeSearch:
    
    @trace_vector_operation("similarity_search")
    def find_similar_nodes(self, query_vector, k=10):
        # 벡터 유사도 검색 추적
        return similar_nodes
    
    @observed("knowledge_search")
    def search(self, query):
        # 전체 검색 과정 추적
        query_vector = self._create_embedding(query)
        candidates = self.find_similar_nodes(query_vector)
        return self._rank_results(candidates)
```

## 7. 개발 환경에서 확인

### 콘솔 출력 확인
초기화 시 `enable_console=True`로 설정하면 콘솔에서 트레이스와 메트릭을 확인할 수 있습니다:

```
Span: document_processing
    - Attributes: {function.name: process_document, function.module: src.domain.services.document_processor}
    - Duration: 245ms
    - Status: OK

Metric: document_processing.calls
    - Value: 1
    - Attributes: {function.name: process_document, status: success}
```

### OTLP 서버 연결
실제 관찰가능성 백엔드(Jaeger, Zipkin, Prometheus 등)에 연결하려면:

```python
initialize_opentelemetry(
    endpoint="http://localhost:4317",  # OTLP 서버 주소
    enable_console=False  # 콘솔 출력 비활성화
)
```

## 8. 환경별 설정 권장사항

### 개발 환경
```python
initialize_opentelemetry(
    enable_console=True,
    enable_tracing=True,
    enable_metrics=True,
    enable_auto_instrumentation=True
)
```

### 프로덕션 환경
```python
initialize_opentelemetry(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    enable_console=False,
    enable_tracing=True,
    enable_metrics=True,
    enable_auto_instrumentation=True
)
```

### 테스트 환경
```python
initialize_opentelemetry(
    enable_console=False,
    enable_tracing=False,
    enable_metrics=False,
    enable_auto_instrumentation=False
)
```

## 9. 성능 고려사항

- **샘플링**: 프로덕션에서는 트레이스 샘플링 설정 고려
- **배치 처리**: BatchSpanProcessor가 기본적으로 사용됨
- **메트릭 수집 간격**: 기본 30초 간격으로 메트릭 전송
- **자동 계측**: HTTP 요청과 SQLite 쿼리가 자동으로 추적됨

## 10. 문제 해결

### OpenTelemetry 패키지가 없는 경우
```bash
uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### 초기화 실패 시
- 환경변수 설정 확인
- 네트워크 연결 확인 (OTLP 서버 사용 시)
- 로그에서 구체적인 오류 메시지 확인

### 트레이스가 보이지 않는 경우
- `initialize_opentelemetry()` 호출 확인
- 데코레이터 적용 확인
- 콘솔 출력 활성화로 디버깅