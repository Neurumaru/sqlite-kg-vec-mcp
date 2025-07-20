"""
ann-benchmarks adapter for sqlite-kg-vec-mcp HNSW implementation.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional

# Add our src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Import external ann-benchmarks base
sys.path.insert(0, str(project_root / "external" / "ann-benchmarks"))

try:
    from ann_benchmarks.algorithms.base.module import BaseANN
except ImportError:
    # Fallback base class if ann-benchmarks is not available
    class BaseANN:
        def __init__(self):
            pass
        
        def get_memory_usage(self):
            import psutil
            return psutil.Process().memory_info().rss / 1024
        
        def fit(self, X):
            pass
        
        def query(self, q, n):
            return []
        
        def done(self):
            pass

from sqlite_kg_vec_mcp.vector.hnsw import HNSWIndex, HNSWBackend


class SqliteKgHNSW(BaseANN):
    """ann-benchmarks adapter for sqlite-kg-vec-mcp HNSW implementation."""
    
    def __init__(self, metric: str, method_param: Dict[str, Any], backend: str = "faiss"):
        """
        Initialize the HNSW adapter.
        
        Args:
            metric: Distance metric ('angular' for cosine, 'euclidean' for l2)
            method_param: Dictionary containing M and efConstruction parameters
            backend: Backend to use ("hnswlib" or "faiss")
        """
        # Convert ann-benchmarks metric names to our format
        self.metric_map = {
            "angular": "cosine", 
            "euclidean": "l2",
            "cosine": "cosine",
            "l2": "l2"
        }
        
        if metric not in self.metric_map:
            raise ValueError(f"Unsupported metric: {metric}. Supported: {list(self.metric_map.keys())}")
        
        self.metric = self.metric_map[metric]
        self.method_param = method_param
        
        # Extract HNSW parameters
        self.M = method_param.get("M", 16)
        self.ef_construction = method_param.get("efConstruction", 200)
        self.ef_search = method_param.get("ef", 50)  # Default ef for search
        
        # Set backend
        if backend == "hnswlib":
            self.backend = HNSWBackend.HNSWLIB
        else:
            self.backend = HNSWBackend.FAISS  # default to FAISS for better performance
        
        # Create temporary directory for index storage
        self.temp_dir = tempfile.mkdtemp(prefix="sqlite_kg_hnsw_")
        
        self.index = None
        self.fitted = False
        self.data_size = 0
        self.dimension = 0
        
        # Name for identification in benchmarks
        self.name = f"sqlite-kg-hnsw-{backend}(M={self.M}, efConstruction={self.ef_construction})"
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the HNSW index to the provided data.
        
        Args:
            X: Training data as numpy array of shape (n_samples, n_features)
        """
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        X = np.asarray(X, dtype=np.float32)
        self.data_size = len(X)
        self.dimension = X.shape[1]
        
        # Initialize HNSW index
        self.index = HNSWIndex(
            space=self.metric,
            dim=self.dimension,
            ef_construction=self.ef_construction,
            M=self.M,
            index_dir=self.temp_dir,
            backend=self.backend
        )
        
        # Initialize the index with known size
        self.index.init_index(max_elements=self.data_size)
        
        # Add all vectors to the index
        for i, vector in enumerate(X):
            self.index.add_item(
                entity_type="benchmark",
                entity_id=i,
                vector=vector
            )
        
        # Set ef for search (can be overridden by set_query_arguments)
        if hasattr(self.index.index, 'set_ef'):
            self.index.index.set_ef(self.ef_search)
        
        self.fitted = True
    
    def set_query_arguments(self, ef: int) -> None:
        """
        Set query-time parameters.
        
        Args:
            ef: ef parameter for search
        """
        if not self.fitted:
            raise ValueError("Index must be fitted before setting query arguments")
        
        self.ef_search = ef
        
        # Set ef parameter if the underlying index supports it
        if hasattr(self.index.index, 'set_ef'):
            self.index.index.set_ef(ef)
        
        # Update name to include ef parameter
        self.name = f"sqlite-kg-hnsw(M={self.M}, efConstruction={self.ef_construction}, ef={ef})"
    
    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """
        Query the index for nearest neighbors.
        
        Args:
            q: Query vector
            n: Number of nearest neighbors to return
            
        Returns:
            Array of indices of nearest neighbors
        """
        if not self.fitted:
            raise ValueError("Index must be fitted before querying")
        
        q = np.asarray(q, dtype=np.float32)
        
        # Ensure query vector has correct dimension
        if len(q) != self.dimension:
            raise ValueError(f"Query vector dimension {len(q)} does not match index dimension {self.dimension}")
        
        # Search using our HNSW implementation
        results = self.index.search(query_vector=q, k=n)
        
        # Extract indices from results
        # results is a list of tuples (distance, entity_id)
        indices = [result[1] for result in results]
        
        return np.array(indices, dtype=np.int32)
    
    def batch_query(self, X: np.ndarray, n: int) -> None:
        """
        Perform batch queries.
        
        Args:
            X: Array of query vectors
            n: Number of nearest neighbors to return for each query
        """
        if not self.fitted:
            raise ValueError("Index must be fitted before querying")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Store results for get_batch_results()
        self.batch_results = []
        
        for query_vector in X:
            result = self.query(query_vector, n)
            self.batch_results.append(result)
    
    def get_batch_results(self) -> np.ndarray:
        """
        Get results from batch_query.
        
        Returns:
            Array of nearest neighbor results for each query
        """
        if not hasattr(self, 'batch_results'):
            raise ValueError("batch_query must be called before get_batch_results")
        
        return np.array(self.batch_results, dtype=object)
    
    def get_memory_usage(self) -> Optional[float]:
        """
        Get current memory usage in kilobytes.
        
        Returns:
            Memory usage in KB, or None if unavailable
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024
        except ImportError:
            return None
    
    def get_additional(self) -> Dict[str, Any]:
        """
        Get additional metadata about the algorithm.
        
        Returns:
            Dictionary of additional attributes
        """
        additional = {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "metric": self.metric,
            "data_size": self.data_size,
            "dimension": self.dimension,
            "implementation": "sqlite-kg-vec-mcp"
        }
        
        # Add index-specific information if available
        if self.index and hasattr(self.index, 'index'):
            try:
                if hasattr(self.index.index, 'get_current_count'):
                    additional["index_size"] = self.index.index.get_current_count()
                if hasattr(self.index.index, 'get_max_elements'):
                    additional["max_elements"] = self.index.index.get_max_elements()
            except:
                pass  # Some methods might not be available
        
        return additional
    
    def done(self) -> None:
        """
        Clean up resources.
        """
        # Clean up index
        if self.index:
            try:
                if hasattr(self.index, 'close'):
                    self.index.close()
            except:
                pass
            finally:
                self.index = None
        
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass  # Best effort cleanup
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.done()
    
    def __str__(self) -> str:
        """String representation."""
        return self.name


# Factory function for easier instantiation
def create_sqlite_kg_hnsw(metric: str, **method_params) -> SqliteKgHNSW:
    """
    Factory function to create SqliteKgHNSW instance.
    
    Args:
        metric: Distance metric
        **method_params: HNSW parameters (M, efConstruction, ef)
        
    Returns:
        SqliteKgHNSW instance
    """
    return SqliteKgHNSW(metric, method_params)