from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import Any, Generic, TypeVar

from policyengine_core.caching import (
    BaseCache,
    FactoryBackedCache,
    InvalidCacheKeyError,
    InvalidCacheValueError,
    RevisionAwareCache,
)

InstantT = TypeVar("InstantT")
ViewT = TypeVar("ViewT")


class ParameterAtInstantCache(
    RevisionAwareCache[InstantT, ViewT, int],
    FactoryBackedCache[InstantT, ViewT],
    BaseCache[InstantT, ViewT],
    MutableMapping[InstantT, ViewT],
    Generic[InstantT, ViewT],
):
    """Cache ordinary parameter views for one policy revision."""

    def __init__(self, *, enabled: bool = True, revision: int = 0) -> None:
        super().__init__()
        self._entries: OrderedDict[InstantT, ViewT] = OrderedDict()
        self._enabled = enabled
        self._revision = revision

    @property
    def cache_enabled(self) -> bool:
        return self._enabled

    @property
    def current_revision(self) -> int:
        return self._revision

    def revision_for_key(self, key: InstantT) -> int:
        self._validate_key(key)
        return self._revision

    def _lookup(self, key: InstantT) -> ViewT:
        return self._entries[key]

    def _store(self, key: InstantT, value: ViewT) -> None:
        self._entries[key] = value

    def _remove(self, key: InstantT) -> bool:
        try:
            del self._entries[key]
        except KeyError:
            return False
        return True

    def _clear_entries(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def _entry_count(self) -> int:
        return len(self._entries)

    def _validate_key(self, key: InstantT) -> None:
        if key is None:
            raise InvalidCacheKeyError("a parameter instant cannot be None")
        try:
            hash(key)
        except TypeError as error:
            raise InvalidCacheKeyError(
                f"an instant key must be hashable, got {type(key).__name__}",
            ) from error

    def _validate_value(self, value: ViewT) -> None:
        if value is None:
            raise InvalidCacheValueError("a parameter view cannot be None")

    def __iter__(self) -> Iterator[InstantT]:
        with self._cache_lock:
            self._ensure_open()
            return iter(tuple(self._entries))

    def __copy__(self) -> ParameterAtInstantCache[InstantT, ViewT]:
        with self._cache_lock:
            self._ensure_open()
            duplicate = type(self)(enabled=self._enabled, revision=self._revision)
            duplicate._entries = self._entries.copy()
            duplicate._cache_peak_entries = len(duplicate._entries)
            return duplicate

    def __deepcopy__(
        self,
        memo: dict[int, Any],
    ) -> ParameterAtInstantCache[InstantT, ViewT]:
        with self._cache_lock:
            self._ensure_open()
            duplicate = type(self)(enabled=self._enabled, revision=self._revision)
            memo[id(self)] = duplicate
            duplicate._entries = copy.deepcopy(self._entries, memo)
            duplicate._cache_peak_entries = len(duplicate._entries)
            return duplicate

    def __getstate__(self) -> dict[str, Any]:
        with self._cache_lock:
            self._ensure_open()
            return {
                "enabled": self._enabled,
                "revision": self._revision,
                "entries": self._entries,
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(enabled=state["enabled"], revision=state["revision"])
        self._entries = state["entries"]
        self._cache_peak_entries = len(self._entries)
