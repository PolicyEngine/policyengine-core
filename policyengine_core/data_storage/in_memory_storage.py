from typing import FrozenSet, Set, Union

from policyengine_core.caching.removed_attribute import RemovedCacheAttribute

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)
from policyengine_core.periods import Period


# What ``InMemoryStorage._inputs`` is while a storage holds no input, for the
# same reason: most storages in a simulation hold no input.
_NO_INPUTS: FrozenSet[str] = frozenset()


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _entry_cache: ImmutableArrayCache[str, ArrayLike]
    _arrays = RemovedCacheAttribute("entry_cache and storage mutation methods")
    _shared = RemovedCacheAttribute("clone with immutable entries")
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
    def entry_cache(self) -> ImmutableArrayCache[str, ArrayLike]:
        """The typed immutable-entry cache owned by this storage."""
        return self._entry_cache

    def __getstate__(self) -> dict:
        return {
            "schema_version": 1,
            "_entry_cache": self._entry_cache,
            "is_eternal": self.is_eternal,
            "_inputs": frozenset(self._inputs),
            "_supplied": frozenset(self._supplied),
        }

    def __setstate__(self, state: dict) -> None:
        expected = {
            "schema_version",
            "_entry_cache",
            "is_eternal",
            "_inputs",
            "_supplied",
        }
        if (
            not isinstance(state, dict)
            or set(state) != expected
            or state["schema_version"] != 1
        ):
            raise ValueError("Unsupported in-memory storage serialization schema")
        cache = state["_entry_cache"]
        inputs, supplied = state["_inputs"], state["_supplied"]
        if not isinstance(cache, ImmutableArrayCache) or not isinstance(
            state["is_eternal"], bool
        ):
            raise ValueError("Invalid in-memory storage cache or eternity flag")
        if not isinstance(inputs, (set, frozenset)) or not isinstance(
            supplied, (set, frozenset)
        ):
            raise ValueError("Invalid in-memory storage provenance")
        entries = cache.entries
        for key, entry in entries.items():
            if (
                not isinstance(key, str)
                or ":" not in key
                or not isinstance(entry, CachedArrayEntry)
            ):
                raise ValueError("Invalid serialized storage entry")
            _, text_period = key.split(":", 1)
            try:
                input_period = periods.period(text_period)
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid serialized storage period") from error
            if str(input_period) != text_period or (
                state["is_eternal"] and input_period.unit != periods.ETERNITY
            ):
                raise ValueError("Serialized storage periods must be canonical")
        keys = entries.keys()
        if not inputs <= keys or not supplied <= inputs:
            raise ValueError("Storage provenance must identify existing input entries")
        self._entry_cache = cache.fork()
        self.is_eternal = state["is_eternal"]
        if inputs:
            self._inputs = frozenset(inputs)
        else:
            self.__dict__.pop("_inputs", None)
        self._supplied = frozenset(supplied)

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
            if inputs <= clone._entry_cache.entries.keys():
                if not isinstance(inputs, frozenset):
                    inputs = self.__dict__["_inputs"] = frozenset(inputs)
                clone._inputs = inputs
            else:
                held = inputs.intersection(clone._entry_cache.entries)
                if held:
                    clone._inputs = frozenset(held)
        return clone

    def get(self, period: Period, branch_name: str = "default") -> ArrayLike:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        key = f"{branch_name}:{period}"
        return self._entry_cache.get_array(key, None)

    def has(self, period: Period, branch_name: str = "default") -> bool:
        """Whether a value is stored for ``period`` under ``branch_name``.

        Unlike ``get``, this never copies an array shared by ``clone``.
        """
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        return f"{branch_name}:{periods.period(period)}" in self._entry_cache

    def is_derived(self, period: Period, branch_name: str = "default") -> bool:
        """Whether the value stored for ``period`` under ``branch_name`` was
        stored with ``derived=True``; ``False`` if none is stored."""
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        key = f"{branch_name}:{periods.period(period)}"
        return key in self._entry_cache and key not in self._inputs

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
        if inputs and not inputs <= self._entry_cache.entries.keys():
            inputs = self._own_inputs()
            inputs.intersection_update(self._entry_cache.entries)
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
            {
                key: self._entry_cache.get(key)
                for key in self._supplied
                if key in self._entry_cache
            }
        )
        self._supplied = self._supplied.intersection(self._entry_cache.entries)
        self._unmark_dropped_keys()

    def delete(self, period: Period = None, branch_name: str = "default") -> None:
        if period is None:
            # Only wipe arrays belonging to the requested branch (previously
            # this wiped every branch regardless of ``branch_name`` — bug C2).
            branch_prefix = f"{branch_name}:"
            self._entry_cache.replace_entries(
                {
                    period_item: value
                    for period_item, value in self._entry_cache.entries.items()
                    if not period_item.startswith(branch_prefix)
                }
            )
            self._unmark_dropped_keys()
            self._supplied = self._supplied.intersection(self._entry_cache.entries)
            return

        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        # Filter by BOTH period containment AND branch_name. Previously the
        # branch_name was silently ignored so deleting a period for one
        # branch deleted it for every branch (bug C2).
        self._entry_cache.replace_entries(
            {
                period_item: value
                for period_item, value in self._entry_cache.entries.items()
                if not (
                    period_item.startswith(f"{branch_name}:")
                    and period.contains(periods.period(period_item.split(":", 1)[1]))
                )
            }
        )
        self._unmark_dropped_keys()
        self._supplied = self._supplied.intersection(self._entry_cache.entries)

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
                self._entry_cache.entries.keys(),
            )
        )

    def get_known_branch_periods(self) -> list:
        return [
            (branch_name, periods.period(period))
            for branch_name, period in map(
                lambda x: x.split(":", 1), self._entry_cache.entries.keys()
            )
        ]

    def get_memory_usage(self) -> dict:
        if not self._entry_cache.entries:
            return dict(
                nb_arrays=0,
                total_nb_bytes=0,
                cell_size=numpy.nan,
            )

        nb_arrays = len(self._entry_cache.entries)
        array = next(iter(self._entry_cache.entries.values()))
        return dict(
            nb_arrays=nb_arrays,
            total_nb_bytes=array.nbytes * nb_arrays,
            cell_size=array.itemsize,
        )
