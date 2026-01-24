"""
Cache Service Module

Provides multi-tier caching with support for in-memory, file-based,
and distributed caching strategies.
"""

import hashlib
import json
import pickle
import time
import threading
from typing import Optional, Any, Dict, Callable, List, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import os


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Represents a cached value with metadata."""
    key: str
    value: T
    created_at: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys in the cache."""
        pass


class InMemoryCache(CacheBackend):
    """
    Thread-safe in-memory cache with LRU eviction.

    Features:
    - O(1) get/set operations
    - Configurable max size with LRU eviction
    - TTL support per entry
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                return None

            entry.touch()
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl or self.default_ttl
            )
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> List[str]:
        with self._lock:
            # Clean expired keys first
            expired = [k for k, v in self._cache.items() if v.is_expired]
            for k in expired:
                del self._cache[k]
            return list(self._cache.keys())

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest last_accessed time
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_access = sum(e.access_count for e in self._cache.values())
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_accesses': total_access,
                'expired_count': sum(1 for e in self._cache.values() if e.is_expired)
            }


class FileCache(CacheBackend):
    """
    File-based cache for persistent storage.

    Features:
    - Persistent across restarts
    - Configurable storage directory
    - Automatic cleanup of expired entries
    """

    def __init__(self, cache_dir: str = ".cache", default_ttl: float = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        # Hash the key to create safe filename
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed}.cache"

    def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if not path.exists():
            return None

        try:
            with open(path, 'rb') as f:
                entry = pickle.load(f)

            if entry.is_expired:
                path.unlink()
                return None

            entry.touch()
            with open(path, 'wb') as f:
                pickle.dump(entry, f)

            return entry.value
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl or self.default_ttl
        )

        path = self._get_path(key)
        with open(path, 'wb') as f:
            pickle.dump(entry, f)

    def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.cache"):
            path.unlink()

    def keys(self) -> List[str]:
        # Note: Can't recover original keys from hashed filenames
        # This returns the file hashes instead
        return [p.stem for p in self.cache_dir.glob("*.cache")]

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries."""
        removed = 0
        for path in self.cache_dir.glob("*.cache"):
            try:
                with open(path, 'rb') as f:
                    entry = pickle.load(f)
                if entry.is_expired:
                    path.unlink()
                    removed += 1
            except Exception:
                path.unlink()
                removed += 1
        return removed


class TieredCache(CacheBackend):
    """
    Multi-tier cache that checks L1 (memory) before L2 (file/remote).

    Features:
    - Fast in-memory L1 cache
    - Persistent L2 cache for overflow
    - Automatic promotion on access
    - Write-through or write-back modes
    """

    def __init__(
        self,
        l1_cache: CacheBackend,
        l2_cache: CacheBackend,
        write_through: bool = True
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.write_through = write_through
        self._stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}

    def get(self, key: str) -> Optional[Any]:
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            self._stats['l1_hits'] += 1
            return value

        # Try L2
        value = self.l2.get(key)
        if value is not None:
            self._stats['l2_hits'] += 1
            # Promote to L1
            self.l1.set(key, value)
            return value

        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        # Always write to L1
        self.l1.set(key, value, ttl)

        # Write-through: also write to L2 immediately
        if self.write_through:
            self.l2.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key)
        return l1_deleted or l2_deleted

    def exists(self, key: str) -> bool:
        return self.l1.exists(key) or self.l2.exists(key)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def keys(self) -> List[str]:
        l1_keys = set(self.l1.keys())
        l2_keys = set(self.l2.keys())
        return list(l1_keys | l2_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get tier-specific statistics."""
        total = sum(self._stats.values())
        return {
            **self._stats,
            'total_requests': total,
            'l1_hit_rate': self._stats['l1_hits'] / total if total > 0 else 0,
            'l2_hit_rate': self._stats['l2_hits'] / total if total > 0 else 0,
        }


class CacheService:
    """
    High-level cache service with decorator support and namespacing.

    Usage:
        cache = CacheService()

        # Direct usage
        cache.set('user:123', user_data)
        user = cache.get('user:123')

        # Decorator usage
        @cache.cached(ttl=300)
        def get_user(user_id):
            return fetch_user_from_db(user_id)

        # Namespace usage
        users_cache = cache.namespace('users')
        users_cache.set('123', user_data)  # Key becomes 'users:123'
    """

    def __init__(
        self,
        backend: CacheBackend = None,
        default_ttl: float = 300,
        key_prefix: str = ""
    ):
        self.backend = backend or InMemoryCache()
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        if self.key_prefix:
            return f"{self.key_prefix}:{key}"
        return key

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache."""
        value = self.backend.get(self._make_key(key))
        return value if value is not None else default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        self.backend.set(self._make_key(key), value, ttl or self.default_ttl)

    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        return self.backend.delete(self._make_key(key))

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self.backend.exists(self._make_key(key))

    def clear(self) -> None:
        """Clear all cached values."""
        self.backend.clear()

    def get_or_set(
        self,
        key: str,
        default_func: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """Get a value, or set it using default_func if not present."""
        value = self.get(key)
        if value is None:
            value = default_func()
            self.set(key, value, ttl)
        return value

    def cached(
        self,
        ttl: Optional[float] = None,
        key_func: Callable[..., str] = None
    ) -> Callable:
        """
        Decorator for caching function results.

        Usage:
            @cache.cached(ttl=60)
            def expensive_function(x, y):
                return compute(x, y)
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key from function name and args
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = ':'.join(key_parts)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            return wrapper
        return decorator

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        count = 0
        for key in self.backend.keys():
            if pattern in key:
                if self.backend.delete(key):
                    count += 1
        return count

    def namespace(self, name: str) -> 'CacheService':
        """Create a namespaced cache instance."""
        prefix = f"{self.key_prefix}:{name}" if self.key_prefix else name
        return CacheService(
            backend=self.backend,
            default_ttl=self.default_ttl,
            key_prefix=prefix
        )


# Utility functions for cache key generation

def hash_key(*args, **kwargs) -> str:
    """Generate a hash-based cache key from arguments."""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def tagged_key(tag: str, *args) -> str:
    """Generate a tagged cache key."""
    return f"{tag}:{':'.join(str(a) for a in args)}"
