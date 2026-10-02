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
import weakref
from typing import Dict, List, Optional, Tuple

from policyengine_core import periods
from policyengine_core.periods import Period

_sequence = itertools.count(1)


def next_sequence_number() -> int:
    """Return a sequence number larger than every one returned before."""
    return next(_sequence)


def advance_sequence_past(number: int) -> None:
    """Make every later sequence number larger than ``number``.

    Numbers come from a counter in each process, so values unpickled from
    another process may carry larger numbers than this process has handed
    out; stores made after them must still be numbered later.
    """
    global _sequence
    _sequence = itertools.count(max(next(_sequence), number + 1))


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
        # Every change to the records since this history was created or last
        # pruned, in order, so a merge takes in only what changed since it
        # last read this history (``period`` is None for a derived record).
        self._journal: List[Tuple[str, Optional[Period], int]] = []
        self._generation = 0  # changes when a prune empties the journal
        # For each history merged into this one, the generation and journal
        # length read last. Keyed weakly: an entry goes with the history it
        # describes (a formula's temporary branch, say).
        self._merged: "weakref.WeakKeyDictionary[StoreHistory, Tuple[int, int]]" = (
            weakref.WeakKeyDictionary()
        )

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        # Weak references do not pickle; merging again only repeats work.
        state["_merged"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        state.setdefault("_journal", [])
        state.setdefault("_generation", 0)
        self.__dict__.update(state)
        self._merged = weakref.WeakKeyDictionary()
        numbers = [
            number
            for stored in self._first_stored.values()
            for number in stored.values()
        ] + list(self._first_derived.values())
        if numbers:
            advance_sequence_past(max(numbers))

    def copy(self) -> "StoreHistory":
        new = StoreHistory()
        new._first_stored = {
            variable_name: dict(stored)
            for variable_name, stored in self._first_stored.items()
        }
        new._first_derived = dict(self._first_derived)
        new._merged = weakref.WeakKeyDictionary(self._merged)
        return new

    def record_store(self, variable_name: str, period: Period, sequence_number: int):
        stored = self._first_stored.setdefault(variable_name, {})
        recorded = stored.get(period)
        if recorded is None or sequence_number < recorded:
            stored[period] = sequence_number
            self._journal.append((variable_name, period, sequence_number))

    def record_derived(self, variable_name: str, sequence_number: int):
        recorded = self._first_derived.get(variable_name)
        if recorded is None or sequence_number < recorded:
            self._first_derived[variable_name] = sequence_number
            self._journal.append((variable_name, None, sequence_number))

    def merge(self, other: "StoreHistory") -> None:
        """Take in ``other``'s records, keeping the earlier number of each."""
        if other is self:
            return
        # Note how far ``other`` had got before reading it: anything it
        # records meanwhile (another thread) is read next time.
        generation, end = other._generation, len(other._journal)
        read = self._merged.get(other)
        if read is not None and read[0] == generation:
            changes = other._journal[read[1] : end]
        else:
            changes = [
                (variable_name, period, sequence_number)
                for variable_name, stored in list(other._first_stored.items())
                for period, sequence_number in list(stored.items())
            ] + [
                (variable_name, None, sequence_number)
                for variable_name, sequence_number in list(other._first_derived.items())
            ]
        for variable_name, period, sequence_number in changes:
            if period is None:
                self.record_derived(variable_name, sequence_number)
            else:
                self.record_store(variable_name, period, sequence_number)
        self._merged[other] = (generation, end)

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
        self._journal = []
        self._generation += 1
        # Merging a history again must restore what was forgotten here.
        self._merged = weakref.WeakKeyDictionary()

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
