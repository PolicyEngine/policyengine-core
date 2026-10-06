from __future__ import annotations

import datetime
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from policyengine_core.caching import (
    BaseCache,
    BoundedCache,
    FactoryBackedCache,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)

SystemT = TypeVar("SystemT")
FrozenCacheValue = tuple[str, Any]


def freeze_cache_value(value: Any) -> FrozenCacheValue:
    """Return an immutable, explicitly typed representation of YAML data."""

    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidCacheKeyError("non-finite floats cannot identify a policy")
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, datetime.date):
        return ("date", value.isoformat())
    if isinstance(value, list):
        return ("list", tuple(freeze_cache_value(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(freeze_cache_value(item) for item in value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise InvalidCacheKeyError("policy mapping keys must be strings")
        return (
            "mapping",
            tuple(
                (key, freeze_cache_value(item)) for key, item in sorted(value.items())
            ),
        )
    raise InvalidCacheKeyError(
        f"unsupported policy cache value type: {type(value).__name__}",
    )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or callable(value):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _reform_identity(reform: Any) -> str:
    if isinstance(reform, str):
        return f"path:{reform}"
    if callable(reform):
        module = getattr(reform, "__module__", type(reform).__module__)
        qualname = getattr(reform, "__qualname__", type(reform).__qualname__)
        return f"callable:{module}.{qualname}"
    raise InvalidCacheKeyError(
        f"unsupported reform identifier type: {type(reform).__name__}",
    )


@dataclass(frozen=True)
class PolicySystemCacheKey:
    """Complete identity of a YAML runner policy-system variant."""

    reforms: tuple[str, ...]
    extensions: tuple[str, ...]
    parameter_overrides: tuple[tuple[str, FrozenCacheValue], ...]

    @classmethod
    def from_inputs(
        cls,
        reforms: Any = (),
        extensions: Any = (),
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> PolicySystemCacheKey:
        reform_items = tuple(_reform_identity(item) for item in _as_sequence(reforms))
        extension_items = _as_sequence(extensions)
        if not all(isinstance(item, str) for item in extension_items):
            raise InvalidCacheKeyError("extension identifiers must be strings")
        overrides = parameter_overrides or {}
        if not isinstance(overrides, Mapping) or not all(
            isinstance(key, str) for key in overrides
        ):
            raise InvalidCacheKeyError("parameter overrides must use string paths")
        frozen_overrides = tuple(
            (path, freeze_cache_value(value))
            for path, value in sorted(overrides.items())
        )
        return cls(
            reforms=reform_items,
            extensions=tuple(extension_items),
            parameter_overrides=frozen_overrides,
        )

    @property
    def is_baseline(self) -> bool:
        return not (self.reforms or self.extensions or self.parameter_overrides)


class PolicySystemCache(
    BoundedCache[PolicySystemCacheKey, SystemT],
    FactoryBackedCache[PolicySystemCacheKey, SystemT],
    BaseCache[PolicySystemCacheKey, SystemT],
    Generic[SystemT],
):
    """Runner-local cache for baseline and derived policy systems."""

    def __init__(
        self,
        max_derived_entries: int = 2,
        *,
        enabled: bool = True,
    ) -> None:
        if max_derived_entries < 0:
            raise ValueError("max_derived_entries must not be negative")
        super().__init__()
        self._systems: OrderedDict[PolicySystemCacheKey, SystemT] = OrderedDict()
        self._max_derived_entries = max_derived_entries
        self._enabled = enabled

    @property
    def cache_enabled(self) -> bool:
        return self._enabled

    @property
    def max_entries(self) -> int:
        return self._max_derived_entries

    def _lookup(self, key: PolicySystemCacheKey) -> SystemT:
        value = self._systems[key]
        self._systems.move_to_end(key)
        return value

    def _store(self, key: PolicySystemCacheKey, value: SystemT) -> None:
        self._systems[key] = value
        self._systems.move_to_end(key)

    def _remove(self, key: PolicySystemCacheKey) -> bool:
        try:
            del self._systems[key]
        except KeyError:
            return False
        return True

    def _clear_entries(self) -> int:
        count = len(self._systems)
        self._systems.clear()
        return count

    def _entry_count(self) -> int:
        return len(self._systems)

    def _bounded_entry_count(self) -> int:
        return sum(not key.is_baseline for key in self._systems)

    def _evict_one(self) -> bool:
        for key in self._systems:
            if not key.is_baseline:
                del self._systems[key]
                return True
        return False

    def _validate_key(self, key: PolicySystemCacheKey) -> None:
        if not isinstance(key, PolicySystemCacheKey):
            raise InvalidCacheKeyError(
                f"expected PolicySystemCacheKey, got {type(key).__name__}",
            )

    def _validate_value(self, value: SystemT) -> None:
        if value is None:
            raise InvalidCacheValueError("a policy system cannot be None")
