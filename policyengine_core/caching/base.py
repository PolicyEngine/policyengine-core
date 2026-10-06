from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from types import TracebackType
from typing import Any, Callable, Generic, TypeVar, overload

K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")
D = TypeVar("D")

_MISSING = object()


class CacheError(RuntimeError):
    """Base exception for cache contract failures."""


class CacheClosedError(CacheError):
    """Raised when an operation is attempted on a closed cache."""


class InvalidCacheKeyError(CacheError, ValueError):
    """Raised when a key does not satisfy a concrete cache contract."""


class InvalidCacheValueError(CacheError, ValueError):
    """Raised when a value does not satisfy a concrete cache contract."""


class StaleCacheRevisionError(CacheError):
    """Raised when a revision-aware cache receives an obsolete key."""


@dataclass(frozen=True)
class CacheInfo:
    """Immutable metrics for a cache instance."""

    hits: int
    misses: int
    writes: int
    builds: int
    deletions: int
    evictions: int
    entries: int
    peak_entries: int
    closed: bool


class BaseCache(ABC, Generic[K, V]):
    """Generic cache lifecycle and metrics implementation.

    Concrete caches provide storage through the protected hooks. Public
    operations are synchronized and must not be overridden merely to change
    the underlying storage representation.
    """

    def __init__(self) -> None:
        self._cache_lock = RLock()
        self._cache_closed = False
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_writes = 0
        self._cache_builds = 0
        self._cache_deletions = 0
        self._cache_evictions = 0
        self._cache_peak_entries = 0

    @abstractmethod
    def _lookup(self, key: K) -> V:
        """Return a stored value or raise ``KeyError``."""

    @abstractmethod
    def _store(self, key: K, value: V) -> None:
        """Insert or replace one value."""

    @abstractmethod
    def _remove(self, key: K) -> bool:
        """Remove one value and report whether it existed."""

    @abstractmethod
    def _clear_entries(self) -> int:
        """Remove every value and return the number removed."""

    @abstractmethod
    def _entry_count(self) -> int:
        """Return the number of retained values."""

    def _validate_key(self, key: K) -> None:
        """Validate a key before any storage operation."""

    def _validate_value(self, value: V) -> None:
        """Validate a value before any storage operation."""

    def _after_write(self, key: K, replaced: bool) -> None:
        """Apply a capability-specific policy after a successful write."""

    def _ensure_open(self) -> None:
        if self._cache_closed:
            raise CacheClosedError(f"{type(self).__name__} is closed")

    def _contains_unmeasured(self, key: K) -> bool:
        try:
            self._lookup(key)
        except KeyError:
            return False
        return True

    @overload
    def get(self, key: K) -> V: ...

    @overload
    def get(self, key: K, default: D) -> V | D: ...

    def get(self, key: K, default: Any = _MISSING) -> Any:
        """Return a cached value or an explicit default."""

        with self._cache_lock:
            self._ensure_open()
            self._validate_key(key)
            try:
                value = self._lookup(key)
            except KeyError:
                self._cache_misses += 1
                if default is _MISSING:
                    raise
                return default
            self._cache_hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        """Insert or replace a value after validating both arguments."""

        with self._cache_lock:
            self._ensure_open()
            self._validate_key(key)
            self._validate_value(value)
            replaced = self._contains_unmeasured(key)
            self._store(key, value)
            self._cache_writes += 1
            self._after_write(key, replaced)
            self._cache_peak_entries = max(
                self._cache_peak_entries,
                self._entry_count(),
            )

    def discard(self, key: K) -> bool:
        """Remove a key if present and report whether it existed."""

        with self._cache_lock:
            self._ensure_open()
            self._validate_key(key)
            removed = self._remove(key)
            if removed:
                self._cache_deletions += 1
            return removed

    def clear(self) -> None:
        """Remove all entries while preserving lifetime metrics."""

        with self._cache_lock:
            self._ensure_open()
            self._cache_deletions += self._clear_entries()

    def close(self) -> None:
        """Release entries and permanently close this cache."""

        with self._cache_lock:
            if self._cache_closed:
                return
            self._cache_deletions += self._clear_entries()
            self._cache_closed = True

    def cache_info(self) -> CacheInfo:
        """Return an immutable snapshot of cache metrics."""

        with self._cache_lock:
            return CacheInfo(
                hits=self._cache_hits,
                misses=self._cache_misses,
                writes=self._cache_writes,
                builds=self._cache_builds,
                deletions=self._cache_deletions,
                evictions=self._cache_evictions,
                entries=self._entry_count(),
                peak_entries=self._cache_peak_entries,
                closed=self._cache_closed,
            )

    def __contains__(self, key: object) -> bool:
        with self._cache_lock:
            self._ensure_open()
            self._validate_key(key)  # type: ignore[arg-type]
            return self._contains_unmeasured(key)  # type: ignore[arg-type]

    def __len__(self) -> int:
        with self._cache_lock:
            self._ensure_open()
            return self._entry_count()

    def __getitem__(self, key: K) -> V:
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        if not self.discard(key):
            raise KeyError(key)

    def __enter__(self) -> BaseCache[K, V]:
        with self._cache_lock:
            self._ensure_open()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class FactoryBackedCache(ABC, Generic[K, V]):
    """Generic capability for caches that construct values on misses."""

    @property
    @abstractmethod
    def cache_enabled(self) -> bool:
        """Whether constructed values are retained."""

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        cache = self
        if not isinstance(cache, BaseCache):
            raise TypeError("FactoryBackedCache requires BaseCache")

        with cache._cache_lock:
            cache._ensure_open()
            cache._validate_key(key)
            if not self.cache_enabled:
                cache._cache_misses += 1
                value = factory()
                cache._validate_value(value)
                cache._cache_builds += 1
                return value

            try:
                return cache.get(key)
            except KeyError:
                value = factory()
                cache.put(key, value)
                cache._cache_builds += 1
                return value


class BoundedCache(ABC, Generic[K, V]):
    """Generic capability for caches with deterministic maximum retention."""

    @property
    @abstractmethod
    def max_entries(self) -> int:
        """Maximum number of retained entries."""

    @abstractmethod
    def _evict_one(self) -> bool:
        """Remove the next entry selected by the concrete eviction policy."""

    def _bounded_entry_count(self) -> int:
        """Return entries governed by ``max_entries``.

        A concrete cache may exclude a distinguished entry, such as a
        baseline policy system, while its ordinary metrics still report every
        retained entry.
        """

        cache = self
        if not isinstance(cache, BaseCache):
            raise TypeError("BoundedCache requires BaseCache")
        return cache._entry_count()

    def _after_write(self, key: K, replaced: bool) -> None:
        cache = self
        if not isinstance(cache, BaseCache):
            raise TypeError("BoundedCache requires BaseCache")
        if self.max_entries < 0:
            raise ValueError("max_entries must not be negative")
        while self._bounded_entry_count() > self.max_entries:
            if not self._evict_one():
                raise CacheError("bounded cache could not evict an entry")
            cache._cache_evictions += 1


class RevisionAwareCache(ABC, Generic[K, V, R]):
    """Generic capability for values bound to a source revision."""

    @property
    @abstractmethod
    def current_revision(self) -> R:
        """Return the source revision currently accepted by the cache."""

    @abstractmethod
    def revision_for_key(self, key: K) -> R:
        """Extract the revision represented by a cache key."""

    def validate_revision(self, key: K) -> None:
        if self.revision_for_key(key) != self.current_revision:
            raise StaleCacheRevisionError(
                f"key revision {self.revision_for_key(key)!r} does not match "
                f"current revision {self.current_revision!r}",
            )


class BranchableCache(ABC, Generic[K, V]):
    """Generic capability for caches with independent branch indexes."""

    @abstractmethod
    def fork(self) -> BranchableCache[K, V]:
        """Return an independent index over safely shareable values."""
