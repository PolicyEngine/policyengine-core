from typing import Dict, Union

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


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _arrays: Dict[Period, ArrayLike]
    is_eternal: bool

    def __init__(self, is_eternal: bool):
        self._arrays = {}
        # Keys of ``_arrays`` whose array still belongs to the storage this
        # one was cloned from with ``share_arrays``. ``get`` replaces each
        # with a copy the first time it is read. A key left here after code
        # outside this class empties ``_arrays`` costs one extra copy at most.
        self._shared = set()
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
            for key, array in self._arrays.items():
                if _can_share(array):
                    clone._arrays[key] = _read_only_view(array)
                    clone._shared.add(key)
                else:
                    clone._arrays[key] = array.copy()
        else:
            clone._arrays = {key: array.copy() for key, array in self._arrays.items()}
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
            self._shared.discard(key)
        return values

    def put(
        self, value: ArrayLike, period: Period, branch_name: str = "default"
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
        self._shared.discard(key)

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
            self._shared.intersection_update(self._arrays)
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
        self._shared.intersection_update(self._arrays)

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
