from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from policyengine_core.caching import (
    BaseCache,
    FactoryBackedCache,
    InvalidCacheKeyError,
    InvalidCacheValueError,
    RevisionAwareCache,
)
from .parameter_revision import ParameterTreeRevision

InstantT = TypeVar("InstantT")
ViewT = TypeVar("ViewT")


@dataclass(frozen=True)
class ParameterAtInstantKey(Generic[InstantT]):
    """Internal cache identity for one policy revision and instant."""

    revision: int
    instant: InstantT


class ParameterAtInstantCache(
    RevisionAwareCache[InstantT, ViewT, int],
    FactoryBackedCache[InstantT, ViewT],
    BaseCache[InstantT, ViewT],
    MutableMapping[InstantT, ViewT],
    Generic[InstantT, ViewT],
):
    """Cache ordinary parameter views for one policy revision."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        revision: int | ParameterTreeRevision = 0,
    ) -> None:
        super().__init__()
        self._entries: OrderedDict[ParameterAtInstantKey[InstantT], ViewT] = (
            OrderedDict()
        )
        self._enabled = enabled
        self._revision_source = (
            revision
            if isinstance(revision, ParameterTreeRevision)
            else ParameterTreeRevision(revision)
        )
        self._observed_revision = self._revision_source.current

    @property
    def cache_enabled(self) -> bool:
        return self._enabled

    @property
    def current_revision(self) -> int:
        return self._revision_source.current

    def revision_for_key(self, key: InstantT) -> int:
        self._validate_key(key)
        return self.current_revision

    @property
    def revision_source(self) -> ParameterTreeRevision:
        return self._revision_source

    def bind_revision(self, revision: ParameterTreeRevision) -> None:
        """Bind the cache to a parameter tree, dropping prior entries."""

        if not isinstance(revision, ParameterTreeRevision):
            raise TypeError("revision must be a ParameterTreeRevision")
        with self._cache_lock:
            self._ensure_open()
            if revision is self._revision_source:
                self._synchronize_revision()
                return
            self._cache_deletions += len(self._entries)
            self._entries.clear()
            self._revision_source = revision
            self._observed_revision = revision.current

    def advance_revision(self) -> int:
        """Advance the bound policy revision and remove obsolete entries."""

        with self._cache_lock:
            self._ensure_open()
            revision = self._revision_source.advance()
            self._synchronize_revision()
            return revision

    def _synchronize_revision(self) -> None:
        current = self._revision_source.current
        if current == self._observed_revision:
            return
        self._cache_deletions += len(self._entries)
        self._entries.clear()
        self._observed_revision = current

    def _composite_key(self, key: InstantT) -> ParameterAtInstantKey[InstantT]:
        return ParameterAtInstantKey(self.current_revision, key)

    def _lookup(self, key: InstantT) -> ViewT:
        self._synchronize_revision()
        return self._entries[self._composite_key(key)]

    def _store(self, key: InstantT, value: ViewT) -> None:
        self._synchronize_revision()
        self._entries[self._composite_key(key)] = value

    def _remove(self, key: InstantT) -> bool:
        self._synchronize_revision()
        try:
            del self._entries[self._composite_key(key)]
        except KeyError:
            return False
        return True

    def _clear_entries(self) -> int:
        self._synchronize_revision()
        count = len(self._entries)
        self._entries.clear()
        return count

    def _entry_count(self) -> int:
        self._synchronize_revision()
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
            self._synchronize_revision()
            return iter(tuple(key.instant for key in self._entries))

    def __copy__(self) -> ParameterAtInstantCache[InstantT, ViewT]:
        with self._cache_lock:
            self._ensure_open()
            duplicate = type(self)(
                enabled=self._enabled,
                revision=self._revision_source.clone(),
            )
            self._synchronize_revision()
            duplicate._entries = self._entries.copy()
            duplicate._cache_peak_entries = len(duplicate._entries)
            return duplicate

    def __deepcopy__(
        self,
        memo: dict[int, Any],
    ) -> ParameterAtInstantCache[InstantT, ViewT]:
        with self._cache_lock:
            self._ensure_open()
            duplicate = type(self)(
                enabled=self._enabled,
                revision=copy.deepcopy(self._revision_source, memo),
            )
            memo[id(self)] = duplicate
            self._synchronize_revision()
            duplicate._entries = copy.deepcopy(self._entries, memo)
            duplicate._cache_peak_entries = len(duplicate._entries)
            return duplicate

    def __getstate__(self) -> dict[str, Any]:
        with self._cache_lock:
            self._ensure_open()
            self._synchronize_revision()
            return {
                "enabled": self._enabled,
                "revision_source": self._revision_source,
                "entries": self._entries,
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        revision = state.get("revision_source", state.get("revision", 0))
        self.__init__(enabled=state["enabled"], revision=revision)
        entries = state["entries"]
        for key, value in entries.items():
            composite = (
                key
                if isinstance(key, ParameterAtInstantKey)
                else ParameterAtInstantKey(self.current_revision, key)
            )
            if composite.revision == self.current_revision:
                self._entries[composite] = value
        self._cache_peak_entries = len(self._entries)
