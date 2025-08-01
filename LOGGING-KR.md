# 로깅 및 예외 처리 가이드라인

이 문서는 SQLite Knowledge Graph Vector MCP 프로젝트에서 헥사고날 아키텍처 원칙을 따르는 로깅 및 예외 처리에 대한 포괄적인 가이드라인을 제공합니다.

## 목차

- [아키텍처 개요](#아키텍처-개요)
- [예외 처리 전략](#예외-처리-전략)
- [로깅 표준](#로깅-표준)
- [계층별 가이드라인](#계층별-가이드라인)
- [모범 사례](#모범-사례)
- [예시](#예시)

## 아키텍처 개요

### 헥사고날 아키텍처 예외 계층

```
Domain Layer (핵심 비즈니스 로직)
    ↑ Domain Exceptions
Port Layer (인터페이스)
    ↑ 변환 & 추상화
Adapter Layer (인프라스트럭처)
    ↑ Infrastructure Exceptions
```

### 의존성 방향
- **Adapters** → **Ports** → **Domain**
- Infrastructure exception은 domain exception을 참조 가능
- Domain exception은 infrastructure exception을 참조하면 안됨

## 예외 처리 전략

### 1. Domain Exceptions (`src/domain/exceptions/`)

**목적**: 비즈니스 규칙 위반 및 도메인 로직 오류 처리

**현재 구조**:
```
src/domain/exceptions/
├── __init__.py
├── base.py                    # DomainException
├── entity_exceptions.py      # 엔티티 관련 비즈니스 오류
├── relationship_exceptions.py # 관계 비즈니스 오류
└── search_exceptions.py      # 검색 비즈니스 오류
```

**사용 시점**:
- 엔티티 검증 실패
- 비즈니스 규칙 위반
- 도메인 제약 위반
- 유효하지 않은 비즈니스 작업

### 2. Infrastructure Exceptions (`src/adapters/exceptions/`)

**목적**: 외부 시스템 및 인프라스트럭처의 기술적 오류 처리

**구조**:
```
src/adapters/exceptions/
├── __init__.py
├── base.py              # InfrastructureException
├── connection.py        # 연결 관련 오류
├── timeout.py          # 타임아웃 관련 오류
├── data.py             # 데이터 처리 오류
└── authentication.py   # 인증/권한 오류
```

**기술별 특화 예외**:
```
src/adapters/[technology]/exceptions.py
├── sqlite3/exceptions.py    # SQLite 특화 오류
├── ollama/exceptions.py     # Ollama 특화 오류
├── vector/exceptions.py     # 벡터 처리 오류
└── fastmcp/exceptions.py    # MCP 프로토콜 오류
```

### 3. 예외 변환 패턴

```python
# Adapter Layer: 기술적 오류를 적절한 예외로 변환
try:
    # 기술적 작업
    cursor.execute(sql, params)
except sqlite3.IntegrityError as e:
    # 도메인 예외로 변환
    raise EntityAlreadyExistsException(entity_id) from e
except sqlite3.OperationalError as e:
    # 인프라스트럭처 예외로 유지
    raise SQLiteConnectionException.from_sqlite_error(db_path, e)
```

## 로깅 표준

### 1. 구조화된 로깅

일관된 필드명으로 구조화된 로깅 사용:

```python
{
    "timestamp": "2024-01-01T12:00:00.123Z",
    "level": "INFO",
    "layer": "domain|port|adapter",
    "component": "entity_service",
    "operation": "create_entity",
    "trace_id": "trace-uuid",
    "span_id": "span-uuid",
    "message": "사람이 읽을 수 있는 메시지",
    "context": {
        "entity_id": "123",
        "entity_type": "Person"
    }
}
```

### 2. 로그 레벨

| 레벨 | 사용법 | 예시 |
|------|--------|------|
| **ERROR** | 시스템 장애, 인프라스트럭처 오류 | 데이터베이스 연결 실패, API 타임아웃 |
| **WARNING** | 비즈니스 규칙 위반, 성능 저하 | 엔티티 검증 실패, 느린 쿼리 |
| **INFO** | 주요 비즈니스 이벤트, API 호출 | 엔티티 생성, 지식 그래프 업데이트 |
| **DEBUG** | 상세 실행 흐름, 개발 정보 | 함수 입출력, 변수 값 |

### 3. 계층별 로깅

#### Domain Layer
```python
# 비즈니스 이벤트에 집중
logger.info("entity_created", 
           entity_id=entity.id, 
           entity_type=entity.type.name)

logger.warning("business_rule_violated", 
               rule="unique_constraint", 
               entity_id=entity.id)
```

#### Adapter Layer
```python
# 인프라스트럭처 이벤트 + 성능에 집중
logger.info("database_query_executed", 
           query_type="select", 
           duration_ms=150,
           table="entities")

logger.error("external_api_failed", 
            service="ollama", 
            endpoint="http://localhost:11434",
            status_code=500, 
            retry_count=3)
```

#### Port Layer
```python
# 계층 전환에 집중
logger.debug("port_method_called", 
            port="EntityRepository", 
            method="save",
            trace_id=trace_id)
```

## 계층별 가이드라인

### Domain Layer

**예외 처리**:
```python
class EntityService:
    def create_entity(self, data: dict) -> Entity:
        try:
            entity = Entity(**data)
            entity.validate()  # InvalidEntityException 발생 가능
            return self.repository.save(entity)
        except InvalidEntityException as e:
            # 도메인 예외 로깅
            logger.warning("entity_validation_failed",
                         error_code=e.error_code,
                         entity_data=data)
            raise  # 도메인 예외 재발생
```

**로깅 포커스**:
- 비즈니스 이벤트
- 규칙 위반
- 엔티티 생명주기 이벤트
- 검색 작업

### Adapter Layer

**예외 처리**:
```python
class SQLiteEntityRepository:
    def save(self, entity: Entity) -> Entity:
        try:
            cursor.execute(sql, params)
            return entity
        except sqlite3.IntegrityError as e:
            # 도메인 예외로 변환
            logger.warning("integrity_constraint_violated",
                         constraint="unique",
                         table="entities",
                         entity_id=entity.id)
            raise EntityAlreadyExistsException(entity.id) from e
        except sqlite3.OperationalError as e:
            # 인프라스트럭처 예외로 유지
            logger.error("database_operation_failed",
                        operation="insert",
                        error=str(e))
            raise SQLiteConnectionException.from_sqlite_error(self.db_path, e)
```

**로깅 포커스**:
- 인프라스트럭처 이벤트
- 성능 메트릭
- 외부 시스템 호출
- 기술적 오류

### Port Layer

**예외 처리**:
```python
class EntityRepositoryPort:
    def save(self, entity: Entity) -> Entity:
        span_id = observability.start_span("repository_save")
        try:
            result = self._concrete_save(entity)
            observability.end_span(span_id, status="success")
            return result
        except DomainException:
            # 도메인 예외는 통과
            observability.end_span(span_id, status="business_error")
            raise
        except InfrastructureException as e:
            # 인프라스트럭처를 도메인 관점으로 변환
            observability.end_span(span_id, status="infrastructure_error")
            raise EntityPersistenceException(f"Save failed: {e}") from e
```

## 모범 사례

### 1. 예외 생성

**✅ 추천**:
```python
# 풍부한 컨텍스트 제공
raise SQLiteIntegrityException(
    constraint="UNIQUE",
    table="entities",
    column="id",
    value=entity_id,
    original_error=sqlite_error
)

# 공통 패턴에 팩토리 메서드 사용
raise SQLiteConnectionException.from_sqlite_error(db_path, sqlite_error)
```

**❌ 비추천**:
```python
# 일반적인 예외는 컨텍스트를 잃음
raise Exception("뭔가 잘못됨")

# Silent failure는 문제를 숨김
except Exception:
    pass
```

### 2. 예외와 함께 로깅

**✅ 추천**:
```python
try:
    result = risky_operation()
    logger.info("operation_succeeded", 
               operation="entity_creation",
               entity_id=result.id)
except DomainException as e:
    logger.warning("domain_error_occurred",
                  error_code=e.error_code,
                  operation="entity_creation",
                  **e.get_context())
    raise
except InfrastructureException as e:
    logger.error("infrastructure_error_occurred",
                error_type=type(e).__name__,
                operation="entity_creation",
                **e.get_context())
    raise
```

### 3. 컨텍스트 보존

**항상 원본 오류 보존**:
```python
try:
    # 원본 작업
    pass
except OriginalError as e:
    # 변환하되 원본 보존
    raise NewException("설명적 메시지", original_error=e) from e
```

### 4. 구조화된 컨텍스트

**의미 있는 컨텍스트 추가**:
```python
exception.add_context("operation", "entity_creation")
exception.add_context("entity_type", "Person")
exception.add_context("retry_count", 3)
```

## 예시

### 완전한 예외 처리 예시

```python
# Domain Service
class EntityService:
    def __init__(self, repository: EntityRepository, logger: Logger):
        self.repository = repository
        self.logger = logger

    def create_entity(self, data: dict) -> Entity:
        operation_id = str(uuid4())
        
        self.logger.info("entity_creation_started",
                        operation_id=operation_id,
                        entity_type=data.get('type'))
        
        try:
            # 도메인 검증
            entity = Entity(**data)
            entity.validate()
            
            # 영속화
            saved_entity = self.repository.save(entity)
            
            self.logger.info("entity_creation_completed",
                           operation_id=operation_id,
                           entity_id=saved_entity.id,
                           duration_ms=get_duration())
            
            return saved_entity
            
        except InvalidEntityException as e:
            self.logger.warning("entity_validation_failed",
                              operation_id=operation_id,
                              error_code=e.error_code,
                              validation_failures=e.get_context())
            raise
            
        except EntityAlreadyExistsException as e:
            self.logger.warning("entity_already_exists",
                              operation_id=operation_id,
                              entity_id=e.entity_id)
            raise
            
        except Exception as e:
            self.logger.error("entity_creation_failed",
                            operation_id=operation_id,
                            error_type=type(e).__name__,
                            error_message=str(e))
            raise
```

### SQLite Adapter 예시

```python
class SQLiteEntityRepository:
    def save(self, entity: Entity) -> Entity:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO entities (id, name, type) VALUES (?, ?, ?)",
                    (entity.id, entity.name, entity.type.value)
                )
                
            logger.info("entity_persisted",
                       entity_id=entity.id,
                       storage="sqlite")
            
            return entity
            
        except sqlite3.IntegrityError as e:
            logger.warning("sqlite_integrity_violation",
                         entity_id=entity.id,
                         constraint="unique",
                         sqlite_error=str(e))
            
            raise EntityAlreadyExistsException(entity.id) from e
            
        except sqlite3.OperationalError as e:
            logger.error("sqlite_operational_error",
                        operation="insert",
                        table="entities",
                        sqlite_error=str(e))
            
            raise SQLiteOperationalException.from_sqlite_error(
                "entity insertion", e, self.db_path
            )
```

### Ollama Adapter 예시

```python
class OllamaClient:
    def generate(self, prompt: str) -> LLMResponse:
        start_time = time.time()
        
        logger.info("llm_generation_started",
                   model=self.model,
                   prompt_length=len(prompt))
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            duration = time.time() - start_time
            
            logger.info("llm_generation_completed",
                       model=self.model,
                       tokens_generated=result.get('eval_count', 0),
                       duration_ms=int(duration * 1000))
            
            return LLMResponse(
                text=result['response'],
                model=self.model,
                tokens_used=result.get('eval_count', 0),
                response_time=duration
            )
            
        except requests.ConnectionError as e:
            logger.error("ollama_connection_failed",
                        base_url=self.base_url,
                        error=str(e))
            
            raise OllamaConnectionException.from_requests_error(
                self.base_url, e
            )
            
        except requests.Timeout as e:
            logger.error("ollama_request_timeout",
                        base_url=self.base_url,
                        timeout_duration=self.timeout)
            
            raise OllamaTimeoutException(
                base_url=self.base_url,
                operation="text generation",
                timeout_duration=self.timeout,
                original_error=e
            )
```

## Observability와의 통합

### 자동 예외 추적

```python
class ObservableLogger:
    def __init__(self, component: str, layer: str):
        self.component = component
        self.layer = layer
        
    def error(self, event: str, **context):
        log_data = {
            "event": event,
            "layer": self.layer,
            "component": self.component,
            "trace_id": get_current_trace_id(),
            "span_id": get_current_span_id(),
            **context
        }
        
        # 구조화된 로깅으로 전송
        structlog.get_logger().error(log_data)
        
        # Observability 백엔드로 전송
        observability.record_metric("error_count", 1, {
            "layer": self.layer,
            "component": self.component,
            "event": event
        })
```

### 예외 메트릭

```python
# 자동 메트릭 수집
def track_exception(exception: Exception, layer: str, component: str):
    metrics.counter("exception_count", {
        "exception_type": type(exception).__name__,
        "layer": layer,
        "component": component
    })
    
    if isinstance(exception, InfrastructureException):
        metrics.counter("infrastructure_error_count", {
            "service": getattr(exception, 'service', 'unknown'),
            "error_code": exception.error_code
        })
```

---

이 로깅 및 예외 처리 전략은 다음을 보장합니다:
- **도메인과 인프라스트럭처 간의 명확한 관심사 분리**
- **디버깅과 모니터링을 위한 풍부한 컨텍스트 보존**
- **모든 어댑터에서 일관된 패턴**
- **프로덕션 모니터링을 위한 관찰 가능한 시스템 동작**
- **유지보수 가능하고 확장 가능한** 오류 처리 아키텍처