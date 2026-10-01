"""The order in which a simulation's values were stored.

Every array a holder stores gets a sequence number from one process-wide
counter, so a larger number means a later store. A value calculated by a
formula is stored after everything the formula read, so it carries a larger
number than each stored value it was calculated from, directly or through
other calculated values.

``Simulation.set_input`` on a branch uses this to drop only the values that
can depend on the input it replaces: see
:meth:`policyengine_core.simulations.Simulation.set_input`.
"""

import itertools
from typing import Dict, Optional

from policyengine_core import periods
from policyengine_core.periods import Period

_sequence = itertools.count(1)


def next_sequence_number() -> int:
    """Return a sequence number larger than every one returned before."""
    return next(_sequence)


def periods_overlap(first: Period, second: Period) -> bool:
    """Whether two periods share at least one day."""
    if periods.ETERNITY in (first.unit, second.unit):
        return True
    return first.start <= second.stop and second.start <= first.stop


class StoreHistory:
    """When each variable's values were first stored in a family of simulations.

    A simulation and the branches created from it (``get_branch``, at any
    depth) share one history. For each variable it records:

    - for each period, the sequence number of the first value stored or
      calculated for that period anywhere in the family, including values a
      holder calculates but does not keep (``variables_to_drop``, the cache
      blacklist), whose readers carry later numbers all the same;
    - the sequence number of the first time a value of the variable was
      derived from its values for other periods: uprated or carried over
      (or given the default because no period it could be uprated or carried
      over from holds a value). Such a value depends on which other periods
      hold values, so an input set for any period can change it.

    Deleting or replacing a stored value does not change the history: a value
    calculated from the deleted or replaced one keeps its number.
    """

    def __init__(self):
        self._first_stored: Dict[str, Dict[Period, int]] = {}
        self._first_derived: Dict[str, int] = {}

    def record_store(self, variable_name: str, period: Period, sequence_number: int):
        stored = self._first_stored.setdefault(variable_name, {})
        if period not in stored:
            stored[period] = sequence_number

    def record_derived(self, variable_name: str, sequence_number: int):
        if variable_name not in self._first_derived:
            self._first_derived[variable_name] = sequence_number

    def earliest_dependency(self, variable_name: str, period: Period) -> Optional[int]:
        """The first sequence number from which a value may depend on ``variable_name`` at ``period``.

        This is the earliest first store of the variable for any period that
        shares a day with ``period``, or the first time one of its values was
        derived from its other periods, whichever came first. ``None`` means
        nothing in the family has read or stored the variable for an
        overlapping period nor derived a value of it, so no stored value can
        depend on its value at ``period``.
        """
        candidates = [
            sequence_number
            for stored_period, sequence_number in self._first_stored.get(
                variable_name, {}
            ).items()
            if periods_overlap(stored_period, period)
        ]
        derived = self._first_derived.get(variable_name)
        if derived is not None:
            candidates.append(derived)
        return min(candidates) if candidates else None
