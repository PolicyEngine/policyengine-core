from typing import Dict, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.periods import Period


def _read_only_view(array: ArrayLike) -> ArrayLike:
    """Return ``array`` as a view that cannot be written through.

    A shared array belongs to the storage it was shared from, so an in-place
    write (``array[mask] = 0``, ``array += 1``) through the sharing storage
    would change the original's value too. The view raises ``ValueError``
    on such a write instead. An array that is already read-only (such as a
    view shared from a further ancestor) is shared as it is.
    """
    if not isinstance(array, numpy.ndarray):
        return array.copy()
    if not array.flags.writeable:
        return array
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
        self.is_eternal = is_eternal

    def clone(self, share_arrays: bool = False) -> "InMemoryStorage":
        """Copy this storage so that writes to either one leave the other unchanged.

        By default every stored array is copied. With ``share_arrays``, the
        clone instead holds a read-only view of each array, so cloning
        allocates no array data. The clone still has its own index: ``put``
        and ``delete`` on either storage replace or drop index entries
        without touching the arrays, so neither storage sees values the
        other stores, replaces or deletes after cloning. Writing into a
        shared array in place through the clone raises ``ValueError``.
        """
        clone = InMemoryStorage(self.is_eternal)
        if share_arrays:
            clone._arrays = {
                key: _read_only_view(array) for key, array in self._arrays.items()
            }
        else:
            clone._arrays = {key: array.copy() for key, array in self._arrays.items()}
        return clone

    def get(self, period: Period, branch_name: str = "default") -> ArrayLike:
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        values = self._arrays.get(f"{branch_name}:{period}")
        if values is None:
            return None
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
        self._arrays[f"{branch_name}:{period}"] = value

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
