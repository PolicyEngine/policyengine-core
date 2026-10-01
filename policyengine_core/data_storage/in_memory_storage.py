from typing import Dict, Optional, Set, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.data_storage.store_history import next_sequence_number
from policyengine_core.periods import Period


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _arrays: Dict[Period, ArrayLike]
    is_eternal: bool

    def __init__(self, is_eternal: bool):
        self._arrays = {}
        # When each array was stored (see ``store_history``), and which were
        # stored as inputs rather than calculated. Both describe the stored
        # value, so ``clone`` copies them with the arrays.
        self._sequence_numbers: Dict[str, int] = {}
        self._input_keys: Set[str] = set()
        self.is_eternal = is_eternal

    def clone(self) -> "InMemoryStorage":
        clone = InMemoryStorage(self.is_eternal)
        clone._arrays = {period: array.copy() for period, array in self._arrays.items()}
        clone._sequence_numbers = dict(self._sequence_numbers)
        clone._input_keys = set(self._input_keys)
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
        self,
        value: ArrayLike,
        period: Period,
        branch_name: str = "default",
        sequence_number: Optional[int] = None,
        is_input: bool = False,
    ) -> None:
        """Store ``value`` for ``period`` on ``branch_name``.

        ``sequence_number`` records when the value was stored (a new number
        by default), and ``is_input`` whether it is an input rather than a
        calculated value; see :meth:`drop_computed`.
        """
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
        self._sequence_numbers[key] = (
            next_sequence_number() if sequence_number is None else sequence_number
        )
        if is_input:
            self._input_keys.add(key)
        else:
            self._input_keys.discard(key)

    def drop_computed(self, since: Optional[int] = None) -> int:
        """Delete stored values that are not inputs, and return how many.

        With ``since``, only values stored with that sequence number or a
        later one are deleted (a value with no recorded number counts as
        later). Inputs, which ``put`` received with ``is_input``, are kept
        whatever their number.
        """
        dropped = [
            key
            for key in self._arrays
            if key not in self._input_keys
            and (since is None or self._sequence_numbers.get(key, since) >= since)
        ]
        for key in dropped:
            del self._arrays[key]
            self._sequence_numbers.pop(key, None)
        return len(dropped)

    def _forget_deleted_keys(self) -> None:
        self._sequence_numbers = {
            key: number
            for key, number in self._sequence_numbers.items()
            if key in self._arrays
        }
        self._input_keys.intersection_update(self._arrays)

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
            self._forget_deleted_keys()
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
        self._forget_deleted_keys()

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
