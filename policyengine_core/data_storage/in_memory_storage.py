from typing import Dict, FrozenSet, Set, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.periods import Period


def _can_share(array: ArrayLike) -> bool:
    """Whether a read-only view of ``array`` protects everything in it.

    A masked array's mask is a second array that a view shares and leaves
    writeable, and anything that is not a numpy array has no views, so both
    are copied straight away instead.
    """
    return isinstance(array, numpy.ndarray) and not isinstance(
        array, numpy.ma.MaskedArray
    )


def _read_only_view(array: numpy.ndarray) -> numpy.ndarray:
    """Return a new view of ``array`` whose data cannot be written through.

    Used for arrays a storage shares with the one it was cloned from, until
    it copies them (see ``InMemoryStorage.clone``). The view is a new array
    object even when ``array`` is already read-only, so that reassigning the
    source's ``shape`` or ``dtype`` does not change what the clone reads.
    """
    view = array.view()
    view.flags.writeable = False
    return view


# What ``InMemoryStorage._shared`` is while a storage shares nothing. Every
# such storage refers to this one object instead of holding an empty set of
# its own: a simulation has a storage for each variable, nearly all of them
# share nothing, and an empty set is 216 bytes.
_NOTHING_SHARED: FrozenSet[str] = frozenset()

# What ``InMemoryStorage._derived`` is while no stored value is derived, for
# the same reason: most storages in a simulation hold no calculated value.
_NOTHING_DERIVED: FrozenSet[str] = frozenset()


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _arrays: Dict[Period, ArrayLike]
    # Keys of ``_arrays`` whose array still belongs to the storage this one
    # was cloned from with ``share_arrays``. ``get`` replaces each with a copy
    # the first time it is read. A storage has a ``_shared`` attribute of its
    # own, a set, only while at least one key is shared; otherwise it reads
    # this class attribute. A key left in the set after code outside this
    # class empties ``_arrays`` costs one extra copy at most.
    _shared: Union[Set[str], FrozenSet[str]] = _NOTHING_SHARED
    # Keys whose value was stored with ``put(..., derived=True)``: calculated
    # by the simulation rather than taken as input. A key counts only while
    # it is stored, and every ``put`` sets or clears its mark. As with
    # ``_shared``, a storage has a set of its own only while at least one
    # stored value is derived; otherwise it reads this class attribute. A
    # storage pickled before derived marks existed reads it too, so its
    # values count as inputs.
    _derived: Union[Set[str], FrozenSet[str]] = _NOTHING_DERIVED
    is_eternal: bool

    def __init__(self, is_eternal: bool):
        self._arrays = {}
        self.is_eternal = is_eternal

    def clone(self, share_arrays: bool = False) -> "InMemoryStorage":
        """Copy this storage.

        By default every stored array is copied straight away.

        With ``share_arrays``, the clone starts with views of this storage's
        arrays and copies an array only when it is first read through
        ``get``, so an array the clone never reads is never copied. What
        ``get`` returns is the clone's own array either way, so writing into
        it in place does not change this storage. (Masked arrays are copied
        straight away: a view would share their mask.)

        The clone has its own index in both cases: ``put`` and ``delete`` on
        either storage replace or drop index entries without touching the
        arrays, so neither storage sees what the other stores, replaces or
        deletes after cloning.

        The one difference from copying straight away: writing in place into
        one of this storage's arrays, after cloning and before the clone
        first reads it, changes the value the clone reads.

        A storage is not safe to read from several threads at once: two first
        reads of the same key can each make a copy.
        """
        clone = InMemoryStorage(self.is_eternal)
        if share_arrays:
            shared = set()
            for key, array in self._arrays.items():
                if _can_share(array):
                    clone._arrays[key] = _read_only_view(array)
                    shared.add(key)
                else:
                    clone._arrays[key] = array.copy()
            if shared:
                clone._shared = shared
        else:
            clone._arrays = {key: array.copy() for key, array in self._arrays.items()}
        # Copy the marks of the keys the clone holds: a mark left behind by
        # code outside this class emptying ``_arrays`` never counts (see
        # ``is_derived``), so a clone has no use for it.
        derived = self._derived
        if derived:
            marks = derived.intersection(clone._arrays)
            if marks:
                clone._derived = marks
        return clone

    def get(self, period: Period, branch_name: str = "default") -> ArrayLike:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        key = f"{branch_name}:{period}"
        values = self._arrays.get(key)
        if values is None:
            return None
        if key in self._shared:
            # First read of an array shared by ``clone``: from here on the
            # caller may write into it, so it needs to be this storage's own.
            values = values.copy()
            self._arrays[key] = values
            self._stop_sharing(key)
        return values

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
        return key in self._derived and key in self._arrays

    # Each method below reads ``self._shared`` once and works on that set, and
    # releases it with one ``dict.pop`` that never raises. A storage is still
    # not safe to read from several threads at once (see ``clone``), but
    # overlapping first reads do not raise here: neither can mutate the
    # shared-nothing object or remove an attribute the other already removed.

    def _stop_sharing(self, key: str) -> None:
        """Record that ``key`` no longer refers to a shared array."""
        shared = self._shared
        if key in shared:
            shared.discard(key)
            self._release_if_empty(shared)

    def _stop_sharing_dropped_keys(self) -> None:
        """Forget the shared keys that ``_arrays`` no longer has."""
        shared = self._shared
        if shared:
            shared.intersection_update(self._arrays)
            self._release_if_empty(shared)

    def _release_if_empty(self, shared: Set[str]) -> None:
        """Go back to the class's shared-nothing object once nothing is shared.

        The storage's own attribute is removed rather than reassigned, so that
        a storage sharing nothing carries none, and a copy or pickle of it
        reads the class attribute too. ``pop`` with a default is one step that
        never raises, so it is safe when another thread has already removed it.
        """
        if not shared:
            self.__dict__.pop("_shared", None)

    def _mark_derived(self, key: str) -> None:
        """Record that the value stored for ``key`` was calculated."""
        derived = self.__dict__.get("_derived")
        if derived is None:
            derived = self.__dict__["_derived"] = set()
        derived.add(key)

    def _unmark_derived(self, key: str) -> None:
        """Record that the value stored for ``key`` is an input."""
        derived = self._derived
        if key in derived:
            derived.discard(key)
            self._release_derived_if_empty(derived)

    def _unmark_dropped_keys(self) -> None:
        """Forget the derived marks of keys ``_arrays`` no longer has."""
        derived = self._derived
        if derived:
            derived.intersection_update(self._arrays)
            self._release_derived_if_empty(derived)

    def _release_derived_if_empty(self, derived: Set[str]) -> None:
        """Go back to the class's nothing-derived object once no stored value
        is derived, as ``_release_if_empty`` does for ``_shared``."""
        if not derived:
            self.__dict__.pop("_derived", None)

    def put(
        self,
        value: ArrayLike,
        period: Period,
        branch_name: str = "default",
        derived: bool = False,
    ) -> None:
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
        self._arrays[key] = value
        self._stop_sharing(key)
        if derived:
            self._mark_derived(key)
        else:
            self._unmark_derived(key)

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
            self._stop_sharing_dropped_keys()
            self._unmark_dropped_keys()
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
        self._stop_sharing_dropped_keys()
        self._unmark_dropped_keys()

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
