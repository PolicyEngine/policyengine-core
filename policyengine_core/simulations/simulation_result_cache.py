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
        self._result_variables: set[str] = set()
        self._input_revision = 0

    @property
    def input_revision(self) -> int:
        """Monotonic revision of this owner's successful supplied-input mutations."""
        return self._input_revision

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

    def supplied_input_keys(self) -> frozenset[SuppliedInputKey[InstantT]]:
        """Return a read-only snapshot of the supplied-input query index."""
        return frozenset(self._supplied_inputs)

    def has_supplied_input(
        self, variable_name: str, branch_name: str, input_period: InstantT
    ) -> bool:
        """Check exact supplied provenance without scanning other inputs."""
        return (
            SuppliedInputKey(variable_name, branch_name, input_period)
            in self._supplied_inputs
        )

    def retain_input_variables(self, variable_names: set[str]) -> None:
        """Prune provenance for removed holders without invalidating retained results."""
        retained = {
            key for key in self._supplied_inputs if key.variable_name in variable_names
        }
        if retained != self._supplied_inputs:
            self._input_revision += 1
        self._supplied_inputs = retained
        self._result_variables.intersection_update(variable_names)

    def record_result_storage(self, variable_name: str) -> None:
        """Record a holder that needs clearing after the next input mutation."""
        self._result_variables.add(variable_name)

    def forget_supplied_input(
        self, variable_name: str, branch_name: str, input_period: InstantT
    ) -> None:
        """Remove exact provenance after replacing an entry, in constant time."""
        key = SuppliedInputKey(variable_name, branch_name, input_period)
        if key in self._supplied_inputs:
            self._supplied_inputs.remove(key)
            self._input_revision += 1

    def take_result_variables(self) -> set[str]:
        """Consume this owner's set of holders containing non-supplied values."""
        names, self._result_variables = self._result_variables, set()
        return names

    def replace_supplied_inputs(
        self,
        entries: set[SuppliedInputKey[InstantT]] | set[tuple[str, str, InstantT]],
    ) -> None:
        normalized = {
            key if isinstance(key, SuppliedInputKey) else SuppliedInputKey(*key)
            for key in entries
        }
        if normalized != self._supplied_inputs:
            self._input_revision += 1
        self._supplied_inputs = normalized

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
        self._validate_input_branch(branch_name)
        self._input_contexts.append(branch_name)
        try:
            yield
        finally:
            self._input_contexts.pop()

    @staticmethod
    def _validate_input_branch(branch_name: str) -> None:
        if not isinstance(branch_name, str) or not branch_name:
            raise InvalidCacheKeyError("input branch names must be non-empty strings")
        if any(separator in branch_name for separator in (":", "/", "\\")):
            raise InvalidCacheKeyError(
                "input branch names cannot contain storage or path separators"
            )

    def record_supplied_input(
        self,
        variable_name: str,
        branch_name: str,
        input_period: InstantT,
    ) -> SuppliedInputKey[InstantT]:
        if not isinstance(variable_name, str) or not variable_name:
            raise InvalidCacheKeyError("input variable names must be non-empty strings")
        self._validate_input_branch(branch_name)
        if input_period is None:
            raise InvalidCacheKeyError("input periods must not be None")
        key = SuppliedInputKey(variable_name, branch_name, input_period)
        self._supplied_inputs.add(key)
        self._input_revision += 1
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
        if removed:
            self._input_revision += 1
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
        self.discard_result_entries(variable_name)
        self._invalidated = {
            key for key in self._invalidated if key[0] != variable_name
        }

    def discard_result_entries(self, variable_name: str) -> None:
        """Evict fast lookups without consuming pending calculation invalidations."""
        for key in tuple(self._entries):
            if key[0] == variable_name:
                self.discard(ResultCacheKey(*key))

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
        clone._result_variables = self._result_variables.copy()
        clone._input_revision = self._input_revision
        return clone

    def fork(self) -> SimulationResultCache[InstantT, ValueT]:
        return self.clone()
