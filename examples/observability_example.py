"""
Example demonstrating the unified logging and observability system.

This example shows how to use the new ObservableLogger with trace context
and structured logging throughout the application.
"""

import asyncio
from typing import Dict, Any

# Import the new observability system
from src.common.observability import (
    get_observable_logger,
    with_observability,
    create_trace_context,
    TraceContextManager,
)
from src.common.logging import configure_structured_logging, LoggingConfig, LogLevel


def setup_logging():
    """Configure structured logging for the application."""
    config = LoggingConfig(
        level=LogLevel.INFO,
        format="json",  # Use JSON format for structured logs
        output="console",
        include_trace=True,
        sanitize_sensitive_data=True
    )
    configure_structured_logging(config)


class ExampleService:
    """Example service demonstrating observability patterns."""
    
    def __init__(self):
        """Initialize service with observable logger."""
        self.logger = get_observable_logger("example_service", "domain")
    
    @with_observability(operation="process_data", include_args=True, include_result=True)
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example method with automatic observability.
        
        The @with_observability decorator automatically:
        - Creates trace context if none exists
        - Logs operation start/completion/failure
        - Measures execution time
        - Handles exceptions with structured logging
        """
        # Simulate some processing
        self.logger.info("data_processing_started", 
                        input_size=len(data),
                        data_keys=list(data.keys()))
        
        try:
            # Simulate processing logic
            result = {
                "processed": True,
                "item_count": len(data),
                "summary": f"Processed {len(data)} items"
            }
            
            # Log intermediate steps
            self.logger.debug("intermediate_step_completed",
                            step="data_validation",
                            validation_result="passed")
            
            return result
            
        except Exception as e:
            # Exception logging is handled automatically by the decorator
            # But you can add additional context if needed
            self.logger.error("data_processing_failed",
                            error_details="Custom error context")
            raise
    
    def manual_trace_example(self, user_id: str) -> None:
        """Example of manual trace context management."""
        # Create trace context manually
        trace_context = create_trace_context(
            operation="user_operation",
            layer="domain",
            component="example_service",
            metadata={"user_id": user_id}
        )
        
        # Use context manager for trace scope
        with TraceContextManager(trace_context):
            self.logger.info("user_operation_started", user_id=user_id)
            
            # All logging within this context will include trace information
            self._step_one(user_id)
            self._step_two(user_id)
            
            self.logger.info("user_operation_completed", user_id=user_id)
    
    def _step_one(self, user_id: str) -> None:
        """Step one with automatic trace context."""
        # Logger automatically includes trace context from parent
        self.logger.debug("step_one_executing", user_id=user_id)
    
    def _step_two(self, user_id: str) -> None:
        """Step two with automatic trace context."""
        self.logger.debug("step_two_executing", user_id=user_id)


class ExampleRepository:
    """Example repository demonstrating adapter-layer logging."""
    
    def __init__(self):
        """Initialize repository with observable logger."""
        self.logger = get_observable_logger("example_repository", "adapter")
    
    @with_observability(operation="save_entity")
    def save_entity(self, entity_data: Dict[str, Any]) -> str:
        """Example save operation with observability."""
        entity_id = entity_data.get("id", "unknown")
        
        # Use operation timing methods
        start_time = self.logger.operation_started("database_save",
                                                 entity_id=entity_id,
                                                 entity_type=entity_data.get("type"))
        
        try:
            # Simulate database operation
            import time
            time.sleep(0.1)  # Simulate DB latency
            
            # Log successful completion
            self.logger.operation_completed("database_save", start_time,
                                          entity_id=entity_id,
                                          table="entities")
            
            return entity_id
            
        except Exception as e:
            # Log operation failure
            self.logger.operation_failed("database_save", start_time, e,
                                       entity_id=entity_id)
            raise
    
    def find_entity(self, entity_id: str) -> Dict[str, Any]:
        """Example find operation with exception handling."""
        try:
            # Simulate database lookup
            if entity_id == "missing":
                # Simulate not found
                raise ValueError("Entity not found")
            
            result = {"id": entity_id, "name": f"Entity {entity_id}"}
            
            self.logger.info("entity_found",
                           entity_id=entity_id,
                           result_size=len(result))
            
            return result
            
        except ValueError as e:
            # Use exception_occurred for rich context
            self.logger.exception_occurred(
                exception=e,
                operation="entity_lookup",
                entity_id=entity_id,
                query_type="by_id"
            )
            raise
        except Exception as e:
            self.logger.exception_occurred(
                exception=e,
                operation="entity_lookup",
                entity_id=entity_id,
                error_category="unexpected"
            )
            raise


async def main():
    """Main example function."""
    # Setup logging
    setup_logging()
    
    # Create service instances
    service = ExampleService()
    repository = ExampleRepository()
    
    print("=== Observability Example ===")
    print("This example demonstrates structured logging with trace context.")
    print("Check the console output for JSON-formatted logs.\n")
    
    # Example 1: Automatic observability with decorator
    print("1. Processing data with @with_observability decorator:")
    try:
        result = service.process_data({"item1": "value1", "item2": "value2"})
        print(f"Result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Manual trace context
    print("2. Manual trace context management:")
    service.manual_trace_example("user_123")
    print()
    
    # Example 3: Repository operations
    print("3. Repository operations with timing:")
    try:
        entity_id = repository.save_entity({
            "id": "ent_456",
            "type": "Person",
            "name": "John Doe"
        })
        print(f"Saved entity: {entity_id}")
        
        entity = repository.find_entity(entity_id)
        print(f"Found entity: {entity}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 4: Error handling
    print("4. Error handling and exception logging:")
    try:
        repository.find_entity("missing")
    except Exception as e:
        print(f"Expected error caught: {e}\n")
    
    print("=== Example Complete ===")
    print("All operations above generated structured logs with trace context.")
    print("In production, these logs would be collected by your logging infrastructure.")


if __name__ == "__main__":
    asyncio.run(main())