# Logging and Exception Handling Guidelines

This document provides comprehensive guidelines for logging and exception handling in the SQLite Knowledge Graph Vector MCP project, following hexagonal architecture principles.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Exception Strategy](#exception-strategy)
- [Logging Standards](#logging-standards)
- [Layer-Specific Guidelines](#layer-specific-guidelines)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Architecture Overview

### Hexagonal Architecture Exception Layers

```
Domain Layer (Core Business Logic)
    ↑ Domain Exceptions
Port Layer (Interfaces)
    ↑ Conversion & Abstraction
Adapter Layer (Infrastructure)
    ↑ Infrastructure Exceptions
```

### Dependency Direction
- **Adapters** → **Ports** → **Domain**
- Infrastructure exceptions can reference domain exceptions
- Domain exceptions should never reference infrastructure exceptions

## Exception Strategy

### 1. Domain Exceptions (`src/domain/exceptions/`)

**Purpose**: Handle business rule violations and domain logic errors.

**Current Structure**:
```
src/domain/exceptions/
├── __init__.py
├── base.py                    # DomainException
├── entity_exceptions.py      # Entity-related business errors
├── relationship_exceptions.py # Relationship business errors
└── search_exceptions.py      # Search business errors
```

**When to Use**:
- Entity validation failures
- Business rule violations
- Domain constraint violations
- Invalid business operations

### 2. Infrastructure Exceptions (`src/adapters/exceptions/`)

**Purpose**: Handle technical errors from external systems and infrastructure.

**Structure**:
```
src/adapters/exceptions/
├── __init__.py
├── base.py              # InfrastructureException
├── connection.py        # Connection-related errors
├── timeout.py          # Timeout-related errors
├── data.py             # Data processing errors
└── authentication.py   # Auth/authorization errors
```

**Technology-Specific Exceptions**:
```
src/adapters/[technology]/exceptions.py
├── sqlite3/exceptions.py    # SQLite-specific errors
├── ollama/exceptions.py     # Ollama-specific errors
├── vector/exceptions.py     # Vector processing errors
└── fastmcp/exceptions.py    # MCP protocol errors
```

### 3. Exception Conversion Pattern

```python
# Adapter Layer: Convert technical errors to appropriate exceptions
try:
    # Technical operation
    cursor.execute(sql, params)
except sqlite3.IntegrityError as e:
    # Convert to domain exception
    raise EntityAlreadyExistsException(entity_id) from e
except sqlite3.OperationalError as e:
    # Keep as infrastructure exception
    raise SQLiteConnectionException.from_sqlite_error(db_path, e)
```

## Logging Standards

### 1. Structured Logging

Use structured logging with consistent field names:

```python
{
    "timestamp": "2024-01-01T12:00:00.123Z",
    "level": "INFO",
    "layer": "domain|port|adapter",
    "component": "entity_service",
    "operation": "create_entity",
    "trace_id": "trace-uuid",
    "span_id": "span-uuid",
    "message": "Human readable message",
    "context": {
        "entity_id": "123",
        "entity_type": "Person"
    }
}
```

### 2. Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| **ERROR** | System failures, infrastructure errors | Database connection failed, API timeout |
| **WARNING** | Business rule violations, degraded performance | Entity validation failed, slow query |
| **INFO** | Major business events, API calls | Entity created, knowledge graph updated |
| **DEBUG** | Detailed execution flow, development info | Function entry/exit, variable values |

### 3. Layer-Specific Logging

#### Domain Layer
```python
# Focus on business events
logger.info("entity_created", 
           entity_id=entity.id, 
           entity_type=entity.type.name)

logger.warning("business_rule_violated", 
               rule="unique_constraint", 
               entity_id=entity.id)
```

#### Adapter Layer
```python
# Focus on infrastructure events + performance
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
# Focus on layer transitions
logger.debug("port_method_called", 
            port="EntityRepository", 
            method="save",
            trace_id=trace_id)
```

## Layer-Specific Guidelines

### Domain Layer

**Exception Handling**:
```python
class EntityService:
    def create_entity(self, data: dict) -> Entity:
        try:
            entity = Entity(**data)
            entity.validate()  # May raise InvalidEntityException
            return self.repository.save(entity)
        except InvalidEntityException as e:
            # Log domain exception
            logger.warning("entity_validation_failed",
                         error_code=e.error_code,
                         entity_data=data)
            raise  # Re-raise domain exception
```

**Logging Focus**:
- Business events
- Rule violations
- Entity lifecycle events
- Search operations

### Adapter Layer

**Exception Handling**:
```python
class SQLiteEntityRepository:
    def save(self, entity: Entity) -> Entity:
        try:
            cursor.execute(sql, params)
            return entity
        except sqlite3.IntegrityError as e:
            # Convert to domain exception
            logger.warning("integrity_constraint_violated",
                         constraint="unique",
                         table="entities",
                         entity_id=entity.id)
            raise EntityAlreadyExistsException(entity.id) from e
        except sqlite3.OperationalError as e:
            # Keep as infrastructure exception
            logger.error("database_operation_failed",
                        operation="insert",
                        error=str(e))
            raise SQLiteConnectionException.from_sqlite_error(self.db_path, e)
```

**Logging Focus**:
- Infrastructure events
- Performance metrics
- External system calls
- Technical errors

### Port Layer

**Exception Handling**:
```python
class EntityRepositoryPort:
    def save(self, entity: Entity) -> Entity:
        span_id = observability.start_span("repository_save")
        try:
            result = self._concrete_save(entity)
            observability.end_span(span_id, status="success")
            return result
        except DomainException:
            # Domain exceptions pass through
            observability.end_span(span_id, status="business_error")
            raise
        except InfrastructureException as e:
            # Convert infrastructure to domain perspective
            observability.end_span(span_id, status="infrastructure_error")
            raise EntityPersistenceException(f"Save failed: {e}") from e
```

## Best Practices

### 1. Exception Creation

**✅ DO**:
```python
# Provide rich context
raise SQLiteIntegrityException(
    constraint="UNIQUE",
    table="entities",
    column="id",
    value=entity_id,
    original_error=sqlite_error
)

# Use factory methods for common patterns
raise SQLiteConnectionException.from_sqlite_error(db_path, sqlite_error)
```

**❌ DON'T**:
```python
# Generic exceptions lose context
raise Exception("Something went wrong")

# Silent failures hide problems
except Exception:
    pass
```

### 2. Logging with Exceptions

**✅ DO**:
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

### 3. Context Preservation

**Always preserve the original error**:
```python
try:
    # Original operation
    pass
except OriginalError as e:
    # Convert but preserve original
    raise NewException("Descriptive message", original_error=e) from e
```

### 4. Structured Context

**Add meaningful context**:
```python
exception.add_context("operation", "entity_creation")
exception.add_context("entity_type", "Person")
exception.add_context("retry_count", 3)
```

## Examples

### Complete Exception Handling Example

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
            # Domain validation
            entity = Entity(**data)
            entity.validate()
            
            # Persistence
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

### SQLite Adapter Example

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

### Ollama Adapter Example

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

## Integration with Observability

### Automatic Exception Tracking

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
        
        # Send to structured logging
        structlog.get_logger().error(log_data)
        
        # Send to observability backend
        observability.record_metric("error_count", 1, {
            "layer": self.layer,
            "component": self.component,
            "event": event
        })
```

### Exception Metrics

```python
# Automatic metrics collection
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

This logging and exception handling strategy ensures:
- **Clear separation of concerns** between domain and infrastructure
- **Rich context preservation** for debugging and monitoring
- **Consistent patterns** across all adapters
- **Observable system behavior** for production monitoring
- **Maintainable and extensible** error handling architecture