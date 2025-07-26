"""
Cache infrastructure port for caching operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import timedelta


class Cache(ABC):
    """
    Secondary port for cache infrastructure operations.

    This interface defines how the domain interacts with caching systems
    for performance optimization and temporary data storage.
    """

    # Basic cache operations
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (seconds or timedelta)

        Returns:
            True if set was successful
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """
        Set expiration time for a key.

        Args:
            key: Cache key
            ttl: Time-to-live (seconds or timedelta)

        Returns:
            True if expiration was set successfully
        """
        pass

    @abstractmethod
    async def ttl(self, key: str) -> Optional[int]:
        """
        Get time-to-live for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, or None if key doesn't exist or has no expiration
        """
        pass

    # Batch operations
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (missing keys are omitted)
        """
        pass

    @abstractmethod
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set multiple values in the cache.

        Args:
            mapping: Dictionary mapping keys to values
            ttl: Time-to-live for all keys

        Returns:
            True if all sets were successful
        """
        pass

    @abstractmethod
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache.

        Args:
            keys: List of cache keys

        Returns:
            Number of keys successfully deleted
        """
        pass

    # Pattern operations
    @abstractmethod
    async def get_keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern.

        Args:
            pattern: Key pattern (supports wildcards like *)

        Returns:
            List of matching keys
        """
        pass

    @abstractmethod
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching a pattern.

        Args:
            pattern: Key pattern (supports wildcards like *)

        Returns:
            Number of keys deleted
        """
        pass

    # Atomic operations
    @abstractmethod
    async def increment(self, key: str, delta: int = 1) -> Optional[int]:
        """
        Atomically increment a numeric value.

        Args:
            key: Cache key
            delta: Increment amount (default: 1)

        Returns:
            New value after increment, or None if key doesn't exist
        """
        pass

    @abstractmethod
    async def decrement(self, key: str, delta: int = 1) -> Optional[int]:
        """
        Atomically decrement a numeric value.

        Args:
            key: Cache key
            delta: Decrement amount (default: 1)

        Returns:
            New value after decrement, or None if key doesn't exist
        """
        pass

    @abstractmethod
    async def add(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Add a value only if the key doesn't exist.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live

        Returns:
            True if value was added, False if key already exists
        """
        pass

    # Hash operations (for structured data)
    @abstractmethod
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """
        Get a field from a hash.

        Args:
            key: Hash key
            field: Field name

        Returns:
            Field value if found, None otherwise
        """
        pass

    @abstractmethod
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """
        Set a field in a hash.

        Args:
            key: Hash key
            field: Field name
            value: Field value

        Returns:
            True if set was successful
        """
        pass

    @abstractmethod
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        Get all fields from a hash.

        Args:
            key: Hash key

        Returns:
            Dictionary of all fields and values
        """
        pass

    @abstractmethod
    async def hmset(self, key: str, mapping: Dict[str, Any]) -> bool:
        """
        Set multiple fields in a hash.

        Args:
            key: Hash key
            mapping: Dictionary mapping field names to values

        Returns:
            True if set was successful
        """
        pass

    @abstractmethod
    async def hdel(self, key: str, field: str) -> bool:
        """
        Delete a field from a hash.

        Args:
            key: Hash key
            field: Field name

        Returns:
            True if deletion was successful
        """
        pass

    # List operations (for ordered data)
    @abstractmethod
    async def lpush(self, key: str, value: Any) -> int:
        """
        Push a value to the left (head) of a list.

        Args:
            key: List key
            value: Value to push

        Returns:
            New length of the list
        """
        pass

    @abstractmethod
    async def rpush(self, key: str, value: Any) -> int:
        """
        Push a value to the right (tail) of a list.

        Args:
            key: List key
            value: Value to push

        Returns:
            New length of the list
        """
        pass

    @abstractmethod
    async def lpop(self, key: str) -> Optional[Any]:
        """
        Pop a value from the left (head) of a list.

        Args:
            key: List key

        Returns:
            Popped value, or None if list is empty
        """
        pass

    @abstractmethod
    async def rpop(self, key: str) -> Optional[Any]:
        """
        Pop a value from the right (tail) of a list.

        Args:
            key: List key

        Returns:
            Popped value, or None if list is empty
        """
        pass

    @abstractmethod
    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        """
        Get a range of values from a list.

        Args:
            key: List key
            start: Start index
            stop: Stop index

        Returns:
            List of values in the specified range
        """
        pass

    @abstractmethod
    async def llen(self, key: str) -> int:
        """
        Get the length of a list.

        Args:
            key: List key

        Returns:
            Length of the list
        """
        pass

    # Cache management
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all data from the cache.

        Returns:
            True if clearing was successful
        """
        pass

    @abstractmethod
    async def flush_expired(self) -> int:
        """
        Remove all expired keys from the cache.

        Returns:
            Number of expired keys removed
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        pass

    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """
        Get cache information and configuration.

        Returns:
            Cache information
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def ping(self) -> bool:
        """
        Ping the cache to check connectivity.

        Returns:
            True if cache responds
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the cache.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Memory usage information
        """
        pass
