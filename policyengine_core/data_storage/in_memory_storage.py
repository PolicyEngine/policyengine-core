from typing import Dict, Union

import numpy
from numpy.typing import ArrayLike

from policyengine_core import periods
from policyengine_core.periods import Period


class InMemoryStorage:
    """
    Low-level class responsible for storing and retrieving calculated vectors in memory
    """

    _arrays: Dict[Period, ArrayLike]
    is_eternal: bool

    def __init__(self, is_eternal: bool):
        self._arrays = {}
        self.is_eternal = is_eternal

    def clone(self) -> "InMemoryStorage":
        clone = InMemoryStorage(self.is_eternal)
        clone._arrays = {
            period: array.copy() for period, array in self._arrays.items()
        }
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

        self._arrays[f"{branch_name}:{period}"] = value

    def delete(
        self, period: Period = None, branch_name: str = "default"
    ) -> None:
        if period is None:
            self._arrays = {}
            return

        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)

        self._arrays = {
            period_item: value
            for period_item, value in self._arrays.items()
            if not period.contains(periods.period(period_item.split(":")[1]))
        }

    def get_known_periods(self) -> list:
        return list(map(lambda x: x.split(":")[1], self._arrays.keys()))

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
