from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike

from policyengine_core.caching import (
    BaseCache,
    BranchableCache,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)

K = TypeVar("K")
ArrayT = TypeVar("ArrayT", bound=ArrayLike)
DefaultT = TypeVar("DefaultT")
_MISSING = object()


def _freeze_array(value: ArrayLike) -> ArrayLike:
    if isinstance(value, np.ma.MaskedArray):
        data_owner = np.array(value.data, copy=True, order="K", subok=False)
        data_owner.flags.writeable = False
        data = data_owner.view()
        data.flags.writeable = False
        if value.mask is np.ma.nomask:
            mask: np.ndarray | np.bool_ = np.ma.nomask
        else:
            mask_owner = np.array(value.mask, copy=True, order="K")
            mask_owner.flags.writeable = False
            mask = mask_owner.view()
            mask.flags.writeable = False
        return np.ma.MaskedArray(
            data,
            mask=mask,
            copy=False,
            fill_value=value.fill_value,
            hard_mask=value.hardmask,
        )
    if isinstance(value, np.ndarray):
        owner = value.copy(order="K")
        owner.flags.writeable = False
        frozen_view = owner.view()
        frozen_view.flags.writeable = False
        return frozen_view
    frozen = copy.deepcopy(value)
    if isinstance(frozen, np.ndarray):
        frozen.flags.writeable = False
    return frozen


def _read_only_view(value: ArrayLike) -> ArrayLike:
    if isinstance(value, np.ma.MaskedArray):
        view = value.view()
        view._data.flags.writeable = False
        if isinstance(view._mask, np.ndarray):
            view._mask.flags.writeable = False
        return view
    if isinstance(value, np.ndarray):
        view = value.view()
        view.flags.writeable = False
        return view
    return copy.deepcopy(value)


def protect_cached_array(value: ArrayT) -> ArrayT:
    """Return a protected value, reusing an already protected cache view."""

    if (
        isinstance(value, np.ndarray)
        and not isinstance(value, np.ma.MaskedArray)
        and not value.flags.writeable
        and isinstance(value.base, np.ndarray)
        and not value.base.flags.writeable
    ):
        return value
    return CachedArrayEntry.from_value(value).read()


@dataclass(frozen=True)
class CachedArrayEntry(Generic[ArrayT]):
    """An immutable owned snapshot of one cached array."""

    _value: ArrayT

    @classmethod
    def from_value(cls, value: ArrayT) -> CachedArrayEntry[ArrayT]:
        return cls(_freeze_array(value))  # type: ignore[arg-type]

    def read(self) -> ArrayT:
        """Return a view that cannot make the owned snapshot writeable."""

        return _read_only_view(self._value)  # type: ignore[return-value]

    @property
    def nbytes(self) -> int:
        return int(getattr(self._value, "nbytes", 0))

    @property
    def itemsize(self) -> float:
        return float(getattr(self._value, "itemsize", np.nan))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        value = np.asarray(self._value, dtype=dtype)
        if copy is True:
            return value.copy()
        return value


class ImmutableArrayCache(
    BaseCache[K, CachedArrayEntry[ArrayT]],
    BranchableCache[K, CachedArrayEntry[ArrayT]],
    Generic[K, ArrayT],
):
    """A generic immutable-value cache with independent forked indexes."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[K, CachedArrayEntry[ArrayT]] = {}

    def _lookup(self, key: K) -> CachedArrayEntry[ArrayT]:
        return self._entries[key]

    def _store(self, key: K, value: CachedArrayEntry[ArrayT]) -> None:
        self._entries[key] = value

    def _remove(self, key: K) -> bool:
        if key not in self._entries:
            return False
        del self._entries[key]
        return True

    def _clear_entries(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def _entry_count(self) -> int:
        return len(self._entries)

    def _validate_key(self, key: K) -> None:
        try:
            hash(key)
        except TypeError as error:
            raise InvalidCacheKeyError("array cache keys must be hashable") from error

    def _validate_value(self, value: CachedArrayEntry[ArrayT]) -> None:
        if not isinstance(value, CachedArrayEntry):
            raise InvalidCacheValueError(
                "immutable array cache values must be CachedArrayEntry values"
            )

    def put_array(self, key: K, value: ArrayT) -> CachedArrayEntry[ArrayT]:
        entry = CachedArrayEntry.from_value(value)
        self.put(key, entry)
        return entry

    @overload
    def get_array(self, key: K) -> ArrayT: ...

    @overload
    def get_array(self, key: K, default: DefaultT) -> ArrayT | DefaultT: ...

    def get_array(self, key: K, default: Any = _MISSING) -> Any:
        if default is _MISSING:
            return self.get(key).read()
        entry = self.get(key, None)
        return default if entry is None else entry.read()

    def entry(self, key: K) -> CachedArrayEntry[ArrayT]:
        return self.get(key)

    @property
    def entries(self) -> dict[K, CachedArrayEntry[ArrayT]]:
        """Mutable compatibility mapping for legacy storage serialization."""

        return self._entries

    def replace_entries(
        self,
        entries: dict[K, CachedArrayEntry[ArrayT] | ArrayT],
    ) -> None:
        normalized = {}
        for key, value in entries.items():
            normalized[key] = (
                value
                if isinstance(value, CachedArrayEntry)
                else CachedArrayEntry.from_value(value)
            )
        self._entries = normalized

    def fork(self) -> ImmutableArrayCache[K, ArrayT]:
        fork: ImmutableArrayCache[K, ArrayT] = ImmutableArrayCache()
        fork._entries = self._entries.copy()
        return fork

    def __getstate__(self) -> dict:
        return {"entries": self._entries}

    def __setstate__(self, state: dict) -> None:
        self.__init__()
        self.replace_entries(state.get("entries", {}))
