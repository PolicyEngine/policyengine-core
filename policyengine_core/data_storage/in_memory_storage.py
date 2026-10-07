from typing import Dict, FrozenSet, Set, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)
from policyengine_core.periods import Period


# Compatibility value for code that inspected the removed copy-on-first-read
# index. Immutable entries make that index unnecessary.
_NOTHING_SHARED: FrozenSet[str] = frozenset()

# What ``InMemoryStorage._inputs`` is while a storage holds no input, for the
# same reason: most storages in a simulation hold no input.
_NO_INPUTS: FrozenSet[str] = frozenset()

# Key a pickled storage's state carries once it records its inputs; a state
# without it comes from a version that recorded neither inputs nor derived
# values (see ``__setstate__``).
_INPUTS_RECORDED = "_inputs_recorded"


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _entry_cache: ImmutableArrayCache[str, ArrayLike]
    # Compatibility view for the former copy-on-first-read index. Immutable
    # entries need no mutable shared-key index.
    _shared: Union[Set[str], FrozenSet[str]] = _NOTHING_SHARED
    # Keys whose value was stored as an input: with ``put(..., derived=False)``,
    # the default. Every other stored value was calculated by the simulation
    # (``put(..., derived=True)``), which ``is_derived`` reports. A key counts
    # only while it is stored, and every ``put`` sets or clears its mark.
    #
    # The storage records inputs rather than derived values because a
    # simulation calculates far more values than it is given (in a
    # policyengine-us household, 5,201 of 5,204 stored values are derived).
    # A storage has a set of its own only while it holds an input; otherwise
    # it reads the class-level empty object.
    #
    # A storage's own inputs are a set it alone changes, or a frozenset it
    # shares with storages cloned from it (or it from them): ``clone`` hands
    # both the same frozenset, and whichever changes its inputs first
    # replaces it with a set of its own (``_own_inputs``). A branch clones
    # every holder, so copying at each clone would cost a set per input
    # holder per branch for inputs most branches never change.
    _inputs: Union[Set[str], FrozenSet[str]] = _NO_INPUTS
    _supplied: Union[Set[str], FrozenSet[str]] = frozenset()
    is_eternal: bool

    def __init__(self, is_eternal: bool):
        self._entry_cache = ImmutableArrayCache()
        self.is_eternal = is_eternal

    @property
    def _arrays(self) -> Dict[str, CachedArrayEntry[ArrayLike]]:
        """Compatibility mapping of storage keys to immutable entries."""

        return self._entry_cache.entries

    @_arrays.setter
    def _arrays(
        self,
        entries: Dict[str, CachedArrayEntry[ArrayLike] | ArrayLike],
    ) -> None:
        cache = getattr(self, "_entry_cache", None)
        if cache is None:
            cache = self._entry_cache = ImmutableArrayCache()
        cache.replace_entries(entries)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state[_INPUTS_RECORDED] = True
        return state

    def __setstate__(self, state: dict) -> None:
        arrays = state.pop("_arrays", None)
        if not state.pop(_INPUTS_RECORDED, False):
            # Pickled before storages recorded their inputs, when nothing
            # marked a calculated value: every stored value counts as an
            # input, as it did then, except any a development version of this
            # change marked derived.
            derived = state.pop("_derived", ())
            cache = state.get("_entry_cache")
            stored_keys = set(arrays if arrays is not None else cache.entries)
            inputs = stored_keys.difference(derived)
            if inputs:
                state["_inputs"] = inputs
        self.__dict__.update(state)
        self._supplied = frozenset(state.get("_supplied", ()))
        if "_entry_cache" not in self.__dict__:
            self._entry_cache = ImmutableArrayCache()
        if arrays is not None:
            self._entry_cache.replace_entries(arrays)

    def clone(self, share_arrays: bool = False) -> "InMemoryStorage":
        """Copy this storage.

        Every clone gets an independent key index over the same immutable
        entries. Reads return protected views, so neither eager copying nor
        copy-on-first-read state is required for isolation. ``share_arrays``
        remains in the signature for country-package compatibility; both
        modes now have the same immutable sharing behavior.
        """
        clone = InMemoryStorage(self.is_eternal)
        clone._entry_cache = self._entry_cache.fork()
        self._supplied = frozenset(self._supplied)
        clone._supplied = self._supplied
        # Share the inputs (see ``_inputs``). A mark left behind by code
        # outside this class emptying ``_arrays`` never counts (see
        # ``is_derived``), so the clone gets only those of keys it holds.
        inputs = self._inputs
        if inputs:
            if inputs <= clone._arrays.keys():
                if not isinstance(inputs, frozenset):
                    inputs = self.__dict__["_inputs"] = frozenset(inputs)
                clone._inputs = inputs
            else:
                held = inputs.intersection(clone._arrays)
                if held:
                    clone._inputs = frozenset(held)
        return clone

    def get(self, period: Period, branch_name: str = "default") -> ArrayLike:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        key = f"{branch_name}:{period}"
        entry = self._arrays.get(key)
        if entry is None:
            return None
        if not isinstance(entry, CachedArrayEntry):
            entry = CachedArrayEntry.from_value(entry)
            self._arrays[key] = entry
        return entry.read()

    def has(self, period: Period, branch_name: str = "default") -> bool:
        """Whether a value is stored for ``period`` under ``branch_name``.

        Unlike ``get``, this never copies an array shared by ``clone``.
        """
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        return f"{branch_name}:{periods.period(period)}" in self._arrays

    def is_derived(self, period: Period, branch_name: str = "default") -> bool:
        """Whether the value stored for ``period`` under ``branch_name`` was
        stored with ``derived=True``; ``False`` if none is stored."""
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        key = f"{branch_name}:{periods.period(period)}"
        return key in self._arrays and key not in self._inputs

    def _own_inputs(self) -> Set[str]:
        """This storage's inputs as a set that only it changes, made from the
        inputs it shares, or from nothing, if need be."""
        inputs = self.__dict__.get("_inputs")
        if inputs is None:
            inputs = self.__dict__["_inputs"] = set()
        elif isinstance(inputs, frozenset):
            inputs = self.__dict__["_inputs"] = set(inputs)
        return inputs

    def _mark_input(self, key: str) -> None:
        """Record that the value stored for ``key`` is an input."""
        if key not in self._inputs:
            self._own_inputs().add(key)

    def _unmark_input(self, key: str) -> None:
        """Record that the value stored for ``key`` was calculated."""
        if key in self._inputs:
            inputs = self._own_inputs()
            inputs.discard(key)
            self._release_inputs_if_empty(inputs)

    def _unmark_dropped_keys(self) -> None:
        """Forget the input marks of keys ``_arrays`` no longer has."""
        inputs = self._inputs
        if inputs and not inputs <= self._arrays.keys():
            inputs = self._own_inputs()
            inputs.intersection_update(self._arrays)
            self._release_inputs_if_empty(inputs)

    def _release_inputs_if_empty(self, inputs: Set[str]) -> None:
        """Go back to the class's no-inputs object once no stored value is an
        input."""
        if not inputs:
            self.__dict__.pop("_inputs", None)

    def put(
        self,
        value: ArrayLike | CachedArrayEntry[ArrayLike],
        period: Period,
        branch_name: str = "default",
        derived: bool = False,
        *,
        supplied: bool = False,
    ) -> CachedArrayEntry[ArrayLike]:
        if supplied and derived:
            raise ValueError("A supplied input cannot also be a derived result")
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        # Keys embed f"{branch_name}:{period}", so reject inputs whose key
        # would be ambiguous to parse back: branch names containing the
        # separator, and month/year periods anchored mid-month (their string
        # form drops the day). See policyengine-core#526 for the structured-
        # key refactor that will lift these restrictions.
        if ":" in branch_name:
            raise ValueError(
                f"Branch name {branch_name!r} may not contain ':' "
                "(see policyengine-core#526)."
            )
        if period.unit in (periods.YEAR, periods.MONTH) and period.start.day != 1:
            raise ValueError(
                f"Cannot cache period {period} anchored mid-month: its "
                "string form is lossy (see policyengine-core#526)."
            )
        key = f"{branch_name}:{period}"
        if isinstance(value, CachedArrayEntry):
            entry = value
            self._entry_cache.put(key, entry)
        else:
            entry = self._entry_cache.put_array(key, value)
        if derived:
            self._unmark_input(key)
        else:
            self._mark_input(key)
        if supplied:
            if not isinstance(self._supplied, set):
                self._supplied = set(self._supplied)
            self._supplied.add(key)
        elif key in self._supplied:
            if not isinstance(self._supplied, set):
                self._supplied = set(self._supplied)
            self._supplied.discard(key)
        return entry

    def is_supplied(self, period: Period, branch_name: str = "default") -> bool:
        """Whether this exact stored entry came from a supported input write."""
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        key = f"{branch_name}:{periods.period(period)}"
        return key in self._supplied and self.has(period, branch_name)

    def retain_supplied_inputs(self) -> None:
        """Drop non-supplied entries without copying any retained payload."""
        self._entry_cache.replace_entries(
            {key: self._arrays[key] for key in self._supplied if key in self._arrays}
        )
        self._supplied = self._supplied.intersection(self._arrays)
        self._unmark_dropped_keys()

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        if period is None:
            # Only wipe arrays belonging to the requested branch (previously
            # this wiped every branch regardless of ``branch_name`` — bug C2).
            branch_prefix = f"{branch_name}:"
            self._arrays = {
                period_item: value
                for period_item, value in self._arrays.items()
                if not period_item.startswith(branch_prefix)
            }
            self._unmark_dropped_keys()
            self._supplied = self._supplied.intersection(self._arrays)
            return

        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        # Filter by BOTH period containment AND branch_name. Previously the
        # branch_name was silently ignored so deleting a period for one
        # branch deleted it for every branch (bug C2).
        self._arrays = {
            period_item: value
            for period_item, value in self._arrays.items()
            if not (
                period_item.startswith(f"{branch_name}:")
                and period.contains(periods.period(period_item.split(":", 1)[1]))
            )
        }
        self._unmark_dropped_keys()
        self._supplied = self._supplied.intersection(self._arrays)

    def discard(self, period: Period, branch_name: str = "default") -> None:
        """Remove one exact key, never its contained or overlapping periods."""
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        key = f"{branch_name}:{periods.period(period)}"
        self._entry_cache.discard(key)
        self._unmark_input(key)
        if key in self._supplied:
            self._supplied = set(self._supplied)
            self._supplied.discard(key)

    def get_known_periods(self) -> list:
        # Split on the first colon only: an anchored period's string form
        # itself contains colons (e.g. "default:year:2027-11").
        return list(
            map(
                lambda x: periods.period(x.split(":", 1)[1]),
                self._arrays.keys(),
            )
        )

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(
                lambda x: x.split(":", 1), self._arrays.keys()
            )
        ]

    def get_memory_usage(self) -> dict:
        if not self._arrays:
            return dict(
                nb_arrays=0,
                total_nb_bytes=0,
                cell_size=numpy.nan,
            )

        nb_arrays = len(self._arrays)
        array = next(iter(self._arrays.values()))
        return dict(
            nb_arrays=nb_arrays,
            total_nb_bytes=array.nbytes * nb_arrays,
            cell_size=array.itemsize,
        )
