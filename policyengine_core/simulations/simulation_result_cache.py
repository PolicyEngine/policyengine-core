from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Generic, NamedTuple, TypeVar

from policyengine_core.caching import (
    BaseCache,
    BranchableCache,
    InvalidCacheKeyError,
)

InstantT = TypeVar("InstantT")
ValueT = TypeVar("ValueT")
_MISSING = object()


class ResultCacheKey(NamedTuple, Generic[InstantT]):
    """A calculated result identified by variable and requested period."""

    variable_name: str
    period: InstantT


class SuppliedInputKey(NamedTuple, Generic[InstantT]):
    """A caller-supplied value identified by variable, branch, and period."""

    variable_name: str
    branch_name: str
    period: InstantT


class SimulationResultCache(
    BaseCache[ResultCacheKey[InstantT], ValueT],
    BranchableCache[ResultCacheKey[InstantT], ValueT],
    Generic[InstantT, ValueT],
):
    """Own simulation-level results, invalidations, and input provenance.

    Holder storage remains responsible for arrays. This class owns the
    indexes that describe how those arrays may be reused or removed, so one
    operation can update calculated results, pending invalidations, and
    supplied-input provenance consistently.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[ResultCacheKey[InstantT], ValueT] = {}
        self._invalidated: set[ResultCacheKey[InstantT]] = set()
        self._supplied_inputs: set[SuppliedInputKey[InstantT]] = set()
        self._input_contexts: list[str] = []

    def _lookup(self, key: ResultCacheKey[InstantT]) -> ValueT:
        return self._entries[key]

    def _store(self, key: ResultCacheKey[InstantT], value: ValueT) -> None:
        self._entries[key] = value

    def _remove(self, key: ResultCacheKey[InstantT]) -> bool:
        return self._entries.pop(key, _MISSING) is not _MISSING

    def _clear_entries(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def _entry_count(self) -> int:
        return len(self._entries)

    def _validate_key(self, key: ResultCacheKey[InstantT]) -> None:
        if not isinstance(key, ResultCacheKey):
            raise InvalidCacheKeyError("result keys must be ResultCacheKey values")
        if not isinstance(key.variable_name, str) or not key.variable_name:
            raise InvalidCacheKeyError(
                "result variable names must be non-empty strings"
            )
        try:
            hash(key.period)
        except TypeError as error:
            raise InvalidCacheKeyError("result periods must be hashable") from error

    @property
    def entries(self) -> dict[ResultCacheKey[InstantT], ValueT]:
        """Return the compatibility mapping used by older country packages."""

        return self._entries

    def replace_entries(
        self,
        entries: dict[ResultCacheKey[InstantT], ValueT]
        | dict[tuple[str, InstantT], ValueT],
    ) -> None:
        """Replace result entries supplied through the legacy attribute."""

        normalized: dict[ResultCacheKey[InstantT], ValueT] = {}
        for key, value in entries.items():
            typed_key = key if isinstance(key, ResultCacheKey) else ResultCacheKey(*key)
            self._validate_key(typed_key)
            normalized[typed_key] = value
        self._entries = normalized

    @property
    def invalidated(self) -> set[ResultCacheKey[InstantT]]:
        return self._invalidated

    def replace_invalidated(
        self,
        entries: set[ResultCacheKey[InstantT]] | set[tuple[str, InstantT]],
    ) -> None:
        self._invalidated = {
            key if isinstance(key, ResultCacheKey) else ResultCacheKey(*key)
            for key in entries
        }

    @property
    def supplied_inputs(self) -> set[SuppliedInputKey[InstantT]]:
        return self._supplied_inputs

    def replace_supplied_inputs(
        self,
        entries: set[SuppliedInputKey[InstantT]] | set[tuple[str, str, InstantT]],
    ) -> None:
        self._supplied_inputs = {
            key if isinstance(key, SuppliedInputKey) else SuppliedInputKey(*key)
            for key in entries
        }

    def replace_input_contexts(self, contexts: list[str]) -> None:
        self._input_contexts = list(contexts)

    @property
    def input_contexts(self) -> list[str]:
        return self._input_contexts

    @property
    def current_input_branch(self) -> str | None:
        return self._input_contexts[-1] if self._input_contexts else None

    @contextmanager
    def supplied_input_context(self, branch_name: str) -> Iterator[None]:
        self._input_contexts.append(branch_name)
        try:
            yield
        finally:
            self._input_contexts.pop()

    def record_supplied_input(
        self,
        variable_name: str,
        branch_name: str,
        input_period: InstantT,
    ) -> SuppliedInputKey[InstantT]:
        if not isinstance(variable_name, str) or not variable_name:
            raise InvalidCacheKeyError("input variable names must be non-empty strings")
        if not isinstance(branch_name, str) or not branch_name:
            raise InvalidCacheKeyError("input branch names must be non-empty strings")
        if input_period is None:
            raise InvalidCacheKeyError("input periods must not be None")
        key = SuppliedInputKey(variable_name, branch_name, input_period)
        self._supplied_inputs.add(key)
        return key

    def supplied_input_periods(
        self,
        variable_name: str,
        visible_branches: tuple[str, ...] | list[str],
    ) -> list[InstantT]:
        visible = set(visible_branches)
        return sorted(
            {
                key[2]
                for key in self._supplied_inputs
                if key[0] == variable_name and key[1] in visible
            },
            key=str,
        )

    @staticmethod
    def _period_contains(container: Any, candidate: Any) -> bool:
        contains = getattr(container, "contains", None)
        return (
            bool(contains(candidate))
            if contains is not None
            else container == candidate
        )

    def discard_supplied_inputs(
        self,
        variable_name: str,
        visible_branches: tuple[str, ...] | list[str],
        deleted_period: InstantT | None = None,
    ) -> set[SuppliedInputKey[InstantT]]:
        visible = set(visible_branches)
        removed = {
            key
            for key in self._supplied_inputs
            if key[0] == variable_name
            and key[1] in visible
            and (
                deleted_period is None or self._period_contains(deleted_period, key[2])
            )
        }
        self._supplied_inputs.difference_update(removed)
        return removed

    def invalidate(self, key: ResultCacheKey[InstantT]) -> None:
        self._validate_key(key)
        self._invalidated.add(key)

    def take_invalidated(self) -> set[ResultCacheKey[InstantT]]:
        invalidated = {
            key if isinstance(key, ResultCacheKey) else ResultCacheKey(*key)
            for key in self._invalidated
        }
        self._invalidated = set()
        return invalidated

    def discard_invalidated(self, key: ResultCacheKey[InstantT]) -> None:
        self._invalidated.discard(key)
        self.discard(key)

    def discard_variable(self, variable_name: str) -> None:
        for key in tuple(self._entries):
            if key[0] == variable_name:
                self.discard(ResultCacheKey(*key))
        self._invalidated = {
            key for key in self._invalidated if key[0] != variable_name
        }

    def clear_calculated_results(self) -> None:
        self.clear()
        self._invalidated.clear()

    def clone(
        self,
        *,
        preserve_results: bool = False,
        preserve_invalidated: bool = False,
    ) -> SimulationResultCache[InstantT, ValueT]:
        clone: SimulationResultCache[InstantT, ValueT] = SimulationResultCache()
        if preserve_results:
            clone._entries = self._entries.copy()
        if preserve_invalidated:
            clone._invalidated = self._invalidated.copy()
        clone._supplied_inputs = self._supplied_inputs.copy()
        return clone

    def fork(self) -> SimulationResultCache[InstantT, ValueT]:
        return self.clone()
