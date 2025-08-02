# SQLite KG Vec MCP

**Document Processing and Knowledge Graph System with Hexagonal Architecture**

A comprehensive knowledge management system that processes documents, extracts knowledge entities and relationships, and provides semantic search capabilities through a clean hexagonal architecture.

*Read this in other languages: [English](README.md), [한국어](README-KR.md)*

## Key Features

- **Document Processing**: Automated extraction of entities and relationships from documents
- **Knowledge Graph**: Store and manage nodes (entities) and relationships with full provenance
- **Semantic Search**: Vector-based similarity search using multiple embedding providers
- **Hexagonal Architecture**: Clean separation of domain logic, ports, and adapters
- **Multiple Adapters**: Support for SQLite, Ollama, OpenAI, HuggingFace, and more
- **MCP Integration**: Model Context Protocol server interface for AI assistant integration

## Installation

```bash
# Clone the repository
git clone https://github.com/Neurumaru/sqlite-kg-vec-mcp.git
cd sqlite-kg-vec-mcp

# Install with pip (including development dependencies)
pip install -e ".[dev]"

# Or use uv for faster dependency resolution
uv pip install -e ".[dev]"
```

## Quick Start

```python
from src.adapters.sqlite3.database import SQLiteDatabase
from src.adapters.ollama.ollama_knowledge_extractor import OllamaKnowledgeExtractor
from src.adapters.openai.text_embedder import OpenAITextEmbedder
from src.domain.services.document_processor import DocumentProcessor
from src.domain.entities.document import Document
from src.domain.value_objects.document_id import DocumentId

# Initialize adapters
database = SQLiteDatabase("knowledge_graph.db")
embedder = OpenAITextEmbedder(api_key="your-openai-key")
knowledge_extractor = OllamaKnowledgeExtractor()

# Initialize domain service with dependency injection
processor = DocumentProcessor(knowledge_extractor)

# Process a document
document = Document(
    id=DocumentId.generate(),
    title="Sample Document",
    content="This is a sample document about artificial intelligence and machine learning."
)

result = await processor.process(document)
print(f"Extracted {result.get_node_count()} nodes and {result.get_relationship_count()} relationships")
```

## Architecture

This project follows **Hexagonal Architecture** (Ports and Adapters) for clean separation of concerns:

```
src/
├── domain/                 # 🏛️ Core business logic (framework-independent)
│   ├── entities/          # Business entities: Document, Node, Relationship
│   ├── value_objects/     # Immutable values: IDs, Vector
│   ├── events/            # Domain events for communication
│   ├── exceptions/        # Domain-specific exceptions
│   └── services/          # Business logic orchestration
├── ports/                 # 🔌 Abstract interfaces (contracts)
│   ├── repositories/      # Data persistence contracts
│   ├── services/          # External service contracts  
│   └── use_cases/         # Application use case contracts
├── adapters/              # 🔧 Technology-specific implementations
│   ├── sqlite3/          # SQLite database adapters
│   ├── ollama/           # Ollama LLM service adapters
│   ├── openai/           # OpenAI service adapters
│   ├── huggingface/      # HuggingFace model adapters
│   ├── hnsw/             # HNSW vector search adapters
│   └── testing/          # Test implementations
├── common/               # 🛠️ Shared utilities
├── tests/                # 🧪 Comprehensive test suite
├── examples/             # 📚 Usage examples
└── docs/                 # 📖 Documentation
```

### Core Benefits

- **🎯 Testability**: Pure domain logic, mockable dependencies
- **🔄 Flexibility**: Swap implementations without changing business logic  
- **🧩 Maintainability**: Clear separation of concerns
- **📈 Scalability**: Easy to add new adapters and services

For detailed architecture explanation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy src
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_graph.py

# Run with coverage
pytest --cov=src
```

## Usage Examples

### Basic Document Processing

```python
from src.adapters.sqlite3.database import SQLiteDatabase
from src.adapters.ollama.ollama_knowledge_extractor import OllamaKnowledgeExtractor
from src.domain.services.document_processor import DocumentProcessor
from src.domain.entities.document import Document
from src.domain.value_objects.document_id import DocumentId

# Initialize components using dependency injection
database = SQLiteDatabase("example.db")
await database.connect()

knowledge_extractor = OllamaKnowledgeExtractor()
processor = DocumentProcessor(knowledge_extractor)

# Process a document
document = Document(
    id=DocumentId.generate(),
    title="AI Research Paper",
    content="Recent advances in machine learning have enabled..."
)

# Extract knowledge
result = await processor.process(document)

print(f"Document processing completed:")
print(f"- Nodes extracted: {result.get_node_count()}")
print(f"- Relationships extracted: {result.get_relationship_count()}")
print(f"- Status: {document.status}")
```

### Working with Different Embedders

```python
from src.adapters.openai.text_embedder import OpenAITextEmbedder
from src.adapters.huggingface.text_embedder import HuggingFaceTextEmbedder
from src.adapters.testing.text_embedder import RandomTextEmbedder

# Using OpenAI embeddings
openai_embedder = OpenAITextEmbedder(api_key="your-key")
vector = await openai_embedder.embed_text("Hello world")
print(f"OpenAI embedding dimension: {openai_embedder.get_embedding_dimension()}")

# Using HuggingFace embeddings
hf_embedder = HuggingFaceTextEmbedder(model_name="all-MiniLM-L6-v2")
vector = await hf_embedder.embed_text("Hello world")
print(f"HuggingFace embedding dimension: {hf_embedder.get_embedding_dimension()}")

# Using test embeddings (for development)
test_embedder = RandomTextEmbedder(dimension=384)
vector = await test_embedder.embed_text("Hello world")
print(f"Test embedding dimension: {test_embedder.get_embedding_dimension()}")
```

### Database Operations

```python
from src.adapters.sqlite3.database import SQLiteDatabase

# Initialize database
db = SQLiteDatabase("knowledge_graph.db")
await db.connect()

# Create tables
schema = {
    "id": "INTEGER PRIMARY KEY",
    "name": "TEXT NOT NULL",
    "properties": "JSON"
}
await db.create_table("entities", schema)

# Execute queries
entities = await db.execute_query(
    "SELECT * FROM entities WHERE name LIKE ?",
    {"name": "%AI%"}
)

# Health check
health = await db.health_check()
print(f"Database status: {health['status']}")
```

### Vector Store Operations

```python
from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.domain.value_objects.vector import Vector

# Initialize vector store
vector_store = SQLiteVectorStore("vectors.db")
await vector_store.connect()
await vector_store.initialize_store(dimension=384)

# Add vectors
vector = Vector([0.1, 0.2, 0.3] * 128)  # 384-dimensional
await vector_store.add_vector("doc1", vector, {"title": "Sample Document"})

# Search for similar vectors
query_vector = Vector([0.2, 0.1, 0.4] * 128)
results = await vector_store.search_similar(query_vector, k=5)

for vector_id, similarity in results:
    print(f"Vector {vector_id}: similarity = {similarity:.4f}")
```

## Design Considerations

The following are key design decisions and considerations for the main components of the system.

### 1. SQLite Schema and Knowledge Graph Modeling

- **Basic Model:** Based on the **Property Graph** model.
- **Entity Tables:**
    - `entities` (or `nodes`): `id` (INTEGER PRIMARY KEY - performance optimization), `uuid` (TEXT UNIQUE NOT NULL - stable identifier for external integration), `name` (TEXT, optional), `type` (TEXT NOT NULL), `properties` (JSON - flexible property storage), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - **Recommended:** Frequently queried 'hot' properties should be separated into **separate columns** for performance, and for specific path queries in JSON, **Generated Columns** and indices on them should be utilized. The `type` field can be considered for separation into a reference table to support consistency and hierarchy.
- **Basic Relationships (Binary):**
    - `edges`: `id` (INTEGER PRIMARY KEY), `source_id` (INTEGER NOT NULL REFERENCES entities(id)), `target_id` (INTEGER NOT NULL REFERENCES entities(id)), `relation_type` (TEXT NOT NULL), `properties` (JSON), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - **Recommended:** The `properties` JSON should be used for storing auxiliary information, and core filtering/join conditions should be in native columns.
- **N-ary Relationship Modeling: Adopting Hyperedges**
    - For complex relationships (three or more participants), the **hyperedge model** is used for consistency.
    - `hyperedges`: `id` (INTEGER PRIMARY KEY), `hyperedge_type` (TEXT NOT NULL), `properties` (JSON - consider integrating metadata and properties), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - `hyperedge_members`: `hyperedge_id` (FK), `entity_id` (FK), `role` (TEXT NOT NULL).
    - **Simplification Option:** Consider integrating related information into the `hyperedges.properties` JSON column instead of using separate `relationship_metadata` and `relationship_properties` tables. This simplifies the schema but requires assessment of query complexity and performance impact. Use generated columns if necessary.
- **Handling Observations (Hybrid Approach):**
    - Frequently accessed or recent information ("hot") is stored in the `entities.properties` JSON field (or separate columns).
    - Older or complete history ("cold") is stored in a separate `observations` table (`id` PK, `entity_id` FK, `content` TEXT, `created_at` DATETIME).
    - **Synchronization:** Implement primarily using **application-level logic** or **periodic batch processing**. Use DB triggers sparingly, considering performance impact. Define data movement criteria (staleness criteria).
- **Indexing:**
    - **Essential:** Create **explicit indices** on `INTEGER PK`, `uuid`, and foreign keys (`source_id`, `target_id`, `hyperedge_id`, `entity_id`).
    - **Recommended:** Create indices on frequently filtered columns such as entity/relationship types (`type`, `relation_type`, `hyperedge_type`) and roles (`hyperedge_members.role`).
    - **Performance Enhancement:** Utilize **Composite Indices** such as `(source_id, relation_type)`, `(target_id, relation_type)` for graph traversal.
    - **JSON Optimization:** Define **Generated Columns** for frequently queried paths in `properties` JSON and create indices on those columns.
    - **Size Optimization:** Consider using **Partial Indices** that index only specific types of data.
- **Database Configuration and Management:**
    - **Basic:** Recommend using `PRAGMA journal_mode=WAL;`, `PRAGMA busy_timeout=5000;`, `PRAGMA synchronous=NORMAL;`.
    - **Additional Recommendations:** Consider `PRAGMA foreign_keys=ON;` (referential integrity), `PRAGMA temp_store=MEMORY;`. Adjust `PRAGMA mmap_size` according to system memory and DB size.
    - **Schema Management:** Consider introducing a `schema_version` table for managing schema change history. Utilize constraints and timestamps actively.

### 2. Vector Storage and Search

- **Storage Method (Separated):** Vector embeddings are stored in separate tables from core graph data.
    - `node_embeddings`: `node_id` (FK to entities.id), `embedding` (BLOB), `model_info` (TEXT or JSON), etc.
    - `relationship_embeddings`: `embedding_id` (PK), `embedding` (BLOB), `model_info` (TEXT or JSON), etc. The ID of this table is linked to hyperedges through `relationship_metadata.embedding_id`.
- **Search Method (Hybrid):**
    - Vector data is permanently stored in SQLite's embedding tables.
    - For fast similarity searches, vector data is loaded into memory to build and use external indices based on **Faiss** or **HNSW** libraries.
    - SQLite IDs are mapped and managed with external vector indices.
- **Embedding Creation and Updates:** Generally **pre-calculated**. Relationship embeddings can be created by combining participating entities and roles. An embedding update strategy is needed for entity/relationship changes.

### 3. Graph Traversal

- **Basic Method:** Implement traversal based on binary relationships (`edges`) and hyperedges (`hyperedge_members` join) using SQLite's **recursive CTE (`WITH RECURSIVE`)**.
- **Optimization:** For performance improvement, cycle detection, depth limitation, and proper index utilization are important. Consider path pre-calculation, adjacency lists, view definitions, etc., if necessary.

### 4. MCP Server Interface

- **Protocol:** Uses **JSON-RPC 2.0** standard message format.
- **Transport:** Uses **WebSocket** as the main transport method.
- **Main API Endpoints:**
    - Entity CRUD
    - Relationship creation/retrieval/modification/deletion (abstracting the complexity of the hyperedge model at the API level)
    - Adding/removing observations
    - Similarity-based search (`search_similar_nodes`, `search_similar_relationships`)
    - Graph traversal (`get_neighbors`, `find_paths`, etc.)
    - Property-based queries
- **Core Role:** The MCP server should **abstract** the complexity of hyperedge-related table (members, metadata, properties) joins and data manipulation to provide clients with a consistent and easy-to-use graph interface.

### 5. Performance Tuning and Scalability Strategy

- **SQLite Optimization:**
    - **Basic:** Use `PRAGMA journal_mode=WAL;`, `PRAGMA busy_timeout=5000;` (set appropriate value), `PRAGMA synchronous=NORMAL;` (consider based on situation), etc. as the default.
    - **Advanced Indexing:** Maximize query performance using partial indices, covering indices, and expression/generated column-based indices.
    - **Query Plan Analysis:** Regularly use `EXPLAIN QUERY PLAN`, `ANALYZE` to identify query performance bottlenecks and improve index design.
    - **Memory/Storage:** Adjust `PRAGMA cache_size`, `PRAGMA temp_store=MEMORY`, etc. according to the system environment. `mmap_size` can help improve read performance but be aware of data corruption risk in case of crash.
    - **KG-Specific Optimization:** When using recursive CTE, apply indexing of related columns and depth limitation, create indices for bidirectional traversal (`(source_id)`, `(target_id)` both), consider using statement cache and application-level cache (LRU).
- **Large Data Processing:**
    - **Realistic Approach:** Avoid arbitrary sharding considering graph data connectivity. Instead, consider **node-centric partitioning** (separation based on node ID/domain) or **full read replication** (using `Litestream`, `rqlite`, etc.) for read scalability.
    - **Limitation Awareness:** Be aware of SQLite single file size and write performance limitations, and consider external distributed system (distributed SQL, graph DB) integration or hybrid architecture if necessary.
- **External Vector Index Performance Management:**
    - **Tuning:** Adjust Faiss/HNSW index parameters (`M`, `efSearch`, `efConstruction`) and consider using memory-efficient index types (e.g., PQ).
    - **Persistence/Backup:** Maintain consistency through index file serialization and version management, **integrated snapshots** with SQLite DB (snapshot simultaneously after stopping writes), WAL checkpoint utilization, cloud storage backup, etc.
    - **Recovery:** Enable detection and reprocessing of outdated embeddings during rollback through embedding version management (using the `embedding_version` column). Testing for recovery scenarios is important.
- **Write Concurrency:**
    - **Basic Strategy:** Use tuned WAL mode and short transactions (grouping related writes for batch processing) as the default.
    - **Additional Measures:** Consider light denormalization for very frequent update paths (e.g., storing counts in node `properties`), and applying the circuit breaker pattern to prevent cascading failures in case of write contention. (Approach app-level queue/sharding with caution due to increased complexity.)

### 6. Data Consistency and Transaction Management

- **SQLite-Vector Index Synchronization:**
    - **Basic Pattern:** Adopt **Asynchronous + Transactional Outbox** pattern as the default. DB transactions focus on relational data, and vector operation intentions are recorded in the Outbox table within the same transaction and then processed asynchronously in a separate process. This separates write performance from indexing operations.
    - **Vector Operations:** Design vector index addition/modification/deletion operations to be **idempotent** to ensure safe retries.
- **Compound Transaction Atomicity:**
    - Use the **Unit of Work** pattern at the application level to process DB changes related to multiple tables as a single SQLite transaction (`BEGIN IMMEDIATE ... COMMIT/ROLLBACK`).
- **Recovery and Consistency Maintenance:**
    - **Failure Handling:** In case of synchronization failure, record detailed information (correlation ID, retry count, error content, etc.) in the **`sync_failures` log table**. A separate recovery service handles this, and if the retry count is exceeded, it is sent to a **Dead-Letter Queue** to prompt manual intervention.
    - **Periodic Reconciliation:** Periodically run a process to check for discrepancies between the DB and vector indices (e.g., comparing embedding versions, comparing hashes) and automatically correct them to ensure eventual consistency.
    - **Rollback:** Rely on idempotent operations and reconciliation processes instead of the Command-Log approach to reduce complexity. Manage related Outbox items to be removed or not processed when DB is rolled back.
- **Selective Synchronous Path:** For a small number of **core operations** (e.g., initial data loading) where immediate consistency is crucial, **limited synchronous processing paths** can be allowed. However, this path should apply **strict timeouts** and **circuit breaker** patterns to minimize the impact on the overall system.
- **Monitoring:** Monitor transaction time, lock waiting, Outbox queue length, `sync_failures` occurrence frequency, etc. to detect bottlenecks and anomalies early.

These design considerations aim to leverage SQLite's relational strengths and performance while integrating flexible knowledge graph representation and vector search capabilities, and providing an effective interface through MCP.

## Dependencies

### Core Dependencies
- **Python 3.8+**: Core runtime
- **SQLite**: Built-in database (no installation required)
- **numpy**: Efficient array operations for vectors

### Adapter Dependencies
- **Text Embeddings**:
  - `sentence-transformers`: HuggingFace embeddings
  - `openai`: OpenAI API embeddings  
  - `torch`: PyTorch framework (for sentence-transformers)
- **LLM Services**:
  - `requests`: HTTP client for Ollama
  - `openai`: OpenAI API client
- **Vector Search**:
  - `hnswlib`: Hierarchical Navigable Small World graphs
- **MCP Integration**:
  - `fastmcp`: Model Context Protocol implementation

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `isort`: Import sorting
- `mypy`: Type checking

All dependencies are optional except core ones - you only install what you need based on which adapters you use.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our coding standards (use Black and isort for formatting, and pass mypy type checking).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [MCP Memory Server Implementation Example (modelcontextprotocol/servers memory)](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)