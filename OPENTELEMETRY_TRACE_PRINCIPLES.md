# OpenTelemetry Tracing 원리 및 Context Propagation

## 1. 핵심 개념들

### Trace ID와 Span ID
```
Trace ID: 0x3b6b45b09d56531d7b50a77c480f6bb9 (128-bit 고유 식별자)
├── Span ID: 0x3550e40a194479fc (루트 스팬)
│   ├── Span ID: 0x4661e50b295480ad (자식 스팬 1)
│   └── Span ID: 0x5772f60c396591be (자식 스팬 2)
```

### Context 구조
```python
SpanContext = {
    "trace_id": "고유한 트레이스 식별자",
    "span_id": "현재 스팬 식별자", 
    "parent_span_id": "부모 스팬 식별자 (있는 경우)",
    "trace_flags": "샘플링 정보 등",
    "trace_state": "벤더별 추가 정보"
}
```

## 2. Context Propagation 메커니즘

### 스레드 로컬 저장소
OpenTelemetry는 Python의 `contextvars`를 사용하여 현재 실행 컨텍스트를 추적합니다:

```python
from contextvars import ContextVar

# OpenTelemetry 내부에서 사용하는 컨텍스트 변수
_CURRENT_SPAN: ContextVar = ContextVar("current_span")

def get_current_span():
    return _CURRENT_SPAN.get(None)

def set_current_span(span):
    _CURRENT_SPAN.set(span)
```

### 실제 동작 방식
```python
# 1. 새로운 스팬 시작
with tracer.start_as_current_span("operation") as span:
    # 2. 현재 스팬이 컨텍스트에 저장됨
    # _CURRENT_SPAN.set(span)
    
    # 3. 중첩된 함수 호출
    nested_function()  # 이 함수는 현재 스팬을 부모로 인식
```

## 3. 자동 계측 (Auto-Instrumentation) 원리

### HTTP 요청 추적
```python
# requests 라이브러리가 자동으로 계측되는 방식
import requests

# 원본 코드
response = requests.get("https://api.example.com")

# 자동 계측된 실제 실행
with tracer.start_as_current_span("HTTP GET") as span:
    span.set_attribute("http.method", "GET")
    span.set_attribute("http.url", "https://api.example.com")
    
    response = original_requests_get("https://api.example.com")
    
    span.set_attribute("http.status_code", response.status_code)
```

### SQLite 쿼리 추적
```python
# SQLite3 쿼리가 자동으로 계측되는 방식
cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))

# 자동 계측된 실제 실행
with tracer.start_as_current_span("SELECT documents") as span:
    span.set_attribute("db.system", "sqlite")
    span.set_attribute("db.statement", "SELECT * FROM documents WHERE id = ?")
    
    result = original_execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    
    span.set_attribute("db.rows_affected", len(result))
```

## 4. 데코레이터 기반 수동 계측

### @traced 데코레이터 내부 동작
```python
@traced("document_processing")
def process_document(doc):
    return processed_doc

# 실제 실행 시:
def process_document(doc):
    tracer = trace.get_tracer(__name__)
    
    # 1. 현재 컨텍스트에서 부모 스팬 확인
    current_span = trace.get_current_span()
    
    # 2. 새로운 스팬 생성 (부모-자식 관계 자동 설정)
    with tracer.start_as_current_span("document_processing") as span:
        # 3. 컨텍스트에 현재 스팬 저장
        # _CURRENT_SPAN.set(span)
        
        # 4. 속성 설정
        span.set_attribute("function.name", "process_document")
        
        # 5. 실제 함수 실행
        result = original_process_document(doc)
        
        # 6. 성공 상태 기록
        span.set_status(Status(StatusCode.OK))
        
        return result
    # 7. 스팬 종료 시 이전 컨텍스트 복원
```

## 5. 분산 시스템에서의 Context Propagation

### HTTP 헤더를 통한 전파
```python
# 발신자 (Client)
with tracer.start_as_current_span("api_call") as span:
    headers = {}
    
    # 현재 스팬 컨텍스트를 HTTP 헤더로 추출
    propagate.inject(headers)
    # headers = {
    #     "traceparent": "00-3b6b45b09d56531d7b50a77c480f6bb9-3550e40a194479fc-01"
    # }
    
    response = requests.get(url, headers=headers)

# 수신자 (Server)  
def handle_request(request):
    # HTTP 헤더에서 스팬 컨텍스트 추출
    parent_context = propagate.extract(request.headers)
    
    # 추출된 컨텍스트를 부모로 하는 새 스팬 생성
    with tracer.start_as_current_span("handle_request", context=parent_context):
        return process_request(request)
```

### traceparent 헤더 형식
```
traceparent: 00-{trace_id}-{parent_span_id}-{flags}
예시: 00-3b6b45b09d56531d7b50a77c480f6bb9-3550e40a194479fc-01

- 00: 버전
- 3b6b45b...6bb9: Trace ID (32자 hex)
- 3550e40...79fc: Parent Span ID (16자 hex)  
- 01: Flags (샘플링 여부 등)
```

## 6. 실제 Request 추적 흐름

### 전체 플로우 예시
```python
# 1. HTTP 요청 시작 (자동 계측)
# Trace ID: abc123... 생성
with auto_span("HTTP POST /documents") as http_span:
    
    # 2. 문서 처리 함수 (수동 계측)
    @traced("document_processing")
    def process_document(doc):
        # 부모: HTTP POST span
        with manual_span("document_processing") as proc_span:
            
            # 3. 엔티티 추출 (수동 계측)
            @trace_llm_operation("gpt-4")
            def extract_entities(text):
                # 부모: document_processing span
                with llm_span("llm.extract_entities") as llm_span:
                    # LLM API 호출 (자동 계측)
                    with auto_span("HTTP POST api.openai.com") as api_span:
                        return call_openai_api(text)
            
            # 4. 데이터베이스 저장 (자동 계측)
            @trace_database_operation("entities")
            def save_entities(entities):
                # 부모: document_processing span
                with db_span("db.save_entities") as db_span:
                    # SQLite 쿼리 (자동 계측)
                    with auto_span("INSERT entities") as sql_span:
                        return cursor.execute("INSERT INTO entities...")
```

### 결과 트레이스 구조
```
Trace ID: abc123...
├── HTTP POST /documents (duration: 2.5s)
    ├── document_processing (duration: 2.3s)
        ├── llm.extract_entities (duration: 1.8s)
        │   └── HTTP POST api.openai.com (duration: 1.7s)
        ├── db.save_entities (duration: 0.3s)
        │   ├── INSERT entities (duration: 0.1s)
        │   ├── INSERT entities (duration: 0.1s)
        │   └── INSERT entities (duration: 0.1s)
        └── vector.create_embeddings (duration: 0.2s)
```

## 7. 성능 최적화 원리

### 배치 처리 (BatchSpanProcessor)
```python
# 스팬들이 즉시 전송되지 않고 배치로 모아서 전송
BatchSpanProcessor(
    exporter=OTLPSpanExporter(),
    max_queue_size=2048,        # 큐 최대 크기
    schedule_delay_millis=5000, # 5초마다 전송
    max_export_batch_size=512   # 한번에 최대 512개 스팬
)
```

### 샘플링 (Sampling)
```python
# 모든 요청을 추적하지 않고 일정 비율만 샘플링
TraceIdRatioBasedSampler(rate=0.1)  # 10%만 추적

# 또는 조건부 샘플링
if is_important_operation(span_name):
    sample_rate = 1.0  # 중요한 작업은 100% 추적
else:
    sample_rate = 0.01  # 일반 작업은 1%만 추적
```

## 8. 메모리 및 가비지 컬렉션

### 스팬 생명주기
```python
# 1. 스팬 생성
span = tracer.start_span("operation")

# 2. 컨텍스트에 저장
token = context.attach(context.set_value("current_span", span))

# 3. 작업 수행 중 속성 추가
span.set_attribute("key", "value")

# 4. 스팬 종료
span.end()

# 5. 컨텍스트에서 제거
context.detach(token)

# 6. BatchSpanProcessor로 전송
processor.on_end(span)

# 7. 메모리에서 해제
del span
```

## 9. 디버깅 및 관찰

### 현재 컨텍스트 확인
```python
from opentelemetry import trace

def debug_current_context():
    current_span = trace.get_current_span()
    if current_span.get_span_context().is_valid():
        print(f"Trace ID: {current_span.get_span_context().trace_id:032x}")
        print(f"Span ID: {current_span.get_span_context().span_id:016x}")
    else:
        print("No active span")

# 함수 내에서 호출하여 현재 추적 상태 확인
@traced("my_operation")
def my_function():
    debug_current_context()  # 추적 정보 출력
```

### 컨텍스트 전파 확인
```python
def trace_propagation_test():
    print("=== Context Propagation Test ===")
    
    with tracer.start_as_current_span("parent") as parent:
        print(f"Parent - Trace: {parent.get_span_context().trace_id:032x}")
        print(f"Parent - Span: {parent.get_span_context().span_id:016x}")
        
        child_function()  # 자식 함수에서도 같은 Trace ID를 가져야 함

@traced("child")
def child_function():
    current = trace.get_current_span()
    print(f"Child - Trace: {current.get_span_context().trace_id:032x}")
    print(f"Child - Span: {current.get_span_context().span_id:016x}")
```

이런 원리로 OpenTelemetry는 각 request의 전체 생명주기를 추적하고, 분산 시스템 전반에서 일관된 관찰가능성을 제공합니다.