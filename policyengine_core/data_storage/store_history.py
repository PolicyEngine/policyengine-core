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
_history_ids = itertools.count(1)


def next_sequence_number() -> int:
    """Return a sequence number larger than every one returned before."""
    return next(_sequence)


def periods_overlap(first: Period, second: Period) -> bool:
    """Whether two periods share at least one day."""
    if periods.ETERNITY in (first.unit, second.unit):
        return True
    return first.start <= second.stop and second.start <= first.stop


class StoreHistory:
    """What the values a simulation holds may have been calculated from.

    Each simulation has its own history. For each variable it records:

    - for each period, the sequence number of the first value stored or
      calculated for that period that a value the simulation holds may have
      been calculated from, including values a holder calculates but does
      not keep (``variables_to_drop``, the cache blacklist), whose readers
      carry later numbers all the same;
    - the sequence number of the first time such a value of the variable was
      derived from its values for other periods: uprated or carried over (or
      given the default because no period it could be uprated or carried over
      from holds a value). Such a value depends on which other periods hold
      values, so an input set for any period can change it.

    The simulation's own stores are recorded as they happen. A branch starts
    with a copy of its parent's history, as it starts with a copy of its
    parent's values. When a formula running in one simulation calculates a
    value in another (a branch it created, for instance), the caller merges
    the other's history into its own, since what the caller stores next may
    be calculated from what it got back. A drop removes the records at or
    after its sequence number (see :meth:`prune`).
    """

    def __init__(self):
        self._first_stored: Dict[str, Dict[Period, int]] = {}
        self._first_derived: Dict[str, int] = {}
        # Changes on every new record, so a merge can skip a history it has
        # already taken in unchanged.
        self._id = next(_history_ids)
        self._version = 0
        self._merged_versions: Dict[int, int] = {}

    def copy(self) -> "StoreHistory":
        new = StoreHistory()
        new._first_stored = {
            variable_name: dict(stored)
            for variable_name, stored in self._first_stored.items()
        }
        new._first_derived = dict(self._first_derived)
        new._merged_versions = dict(self._merged_versions)
        return new

    def record_store(self, variable_name: str, period: Period, sequence_number: int):
        stored = self._first_stored.setdefault(variable_name, {})
        recorded = stored.get(period)
        if recorded is None or sequence_number < recorded:
            stored[period] = sequence_number
            self._version += 1

    def record_derived(self, variable_name: str, sequence_number: int):
        recorded = self._first_derived.get(variable_name)
        if recorded is None or sequence_number < recorded:
            self._first_derived[variable_name] = sequence_number
            self._version += 1

    def merge(self, other: "StoreHistory") -> None:
        """Take in ``other``'s records, keeping the earlier number of each."""
        if other is self or self._merged_versions.get(other._id) == other._version:
            return
        for variable_name, stored in other._first_stored.items():
            for period, sequence_number in stored.items():
                self.record_store(variable_name, period, sequence_number)
        for variable_name, sequence_number in other._first_derived.items():
            self.record_derived(variable_name, sequence_number)
        self._merged_versions[other._id] = other._version

    def prune(self, since: Optional[int] = None) -> None:
        """Forget the records numbered ``since`` or later (all, without ``since``).

        Called after the simulation drops every value it holds, other than
        inputs, numbered ``since`` or later: nothing it still holds was
        calculated from what those records describe. The caller records again
        the inputs it keeps.
        """
        self._first_stored = {
            variable_name: kept
            for variable_name, stored in self._first_stored.items()
            if (
                kept := {
                    period: sequence_number
                    for period, sequence_number in stored.items()
                    if since is not None and sequence_number < since
                }
            )
        }
        self._first_derived = {
            variable_name: sequence_number
            for variable_name, sequence_number in self._first_derived.items()
            if since is not None and sequence_number < since
        }
        self._version += 1
        # Merging a history again must restore what was forgotten here.
        self._merged_versions = {}

    def earliest_dependency(self, variable_name: str, period: Period) -> Optional[int]:
        """The first sequence number from which a value may depend on ``variable_name`` at ``period``.

        This is the earliest recorded store of the variable for any period
        that shares a day with ``period``, or the earliest recorded derivation
        of one of its values from other periods, whichever came first.
        ``None`` means no value the simulation holds can depend on the
        variable's value at ``period``.
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
