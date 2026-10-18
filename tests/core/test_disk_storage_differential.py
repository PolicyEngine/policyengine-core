"""Storing holder values on disk gives the results storing them in memory gives.

A Hypothesis property over random sequences of branching, cloning, inputs,
calculations, derivatives, reforms, holder replacement and drops, run on a
disk-backed and a memory-backed simulation family side by side.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.simulations import Simulation
from tests.fixtures.disk_storage import build_simulation, unraisable_exceptions

MONTHS = ["2017-01", "2017-02"]
NAMES = ["a", "b", "c"]
INPUTS = ["salary", "rent", "housing_occupancy_status", "birth"]
CALCULATED = [
    "salary",
    "income_tax",
    "disposable_income",
    "basic_income",
    "housing_allowance",
    "housing_occupancy_status",
    "total_taxes",
    "household_income",
]
HOUSING_STATUSES = ["owner", "tenant", "free_lodger", "homeless"]
COUNT = 2


def _input_value(variable: str, seed: int) -> np.ndarray:
    values = [seed, seed // 7 + 1]
    if variable == "housing_occupancy_status":
        return np.array([HOUSING_STATUSES[value % 4] for value in values])
    if variable == "birth":
        return np.array(
            [f"{1950 + value % 60}-01-01" for value in values], dtype="datetime64[D]"
        )
    return np.array(values, dtype=float)


@dataclass
class _Member:
    simulation: Simulation
    parent: Optional["_Member"]
    branch_name: Optional[str]


class _Family:
    """A simulation and every branch or clone made from it, still in use."""

    def __init__(self, on_disk: bool):
        self.members = [_Member(build_simulation(on_disk, count=COUNT), None, None)]

    def pick(self, index: int) -> _Member:
        return self.members[index % len(self.members)]

    def apply(self, operation: tuple):
        kind, index, *arguments = operation
        member = self.pick(index)
        simulation = member.simulation
        if kind == "branch":
            (name,) = arguments
            branch = simulation.get_branch(name)
            if not any(other.simulation is branch for other in self.members):
                self.members.append(_Member(branch, member, name))
        elif kind == "clone":
            self.members.append(_Member(simulation.clone(), member, None))
        elif kind == "set":
            variable, period, seed = arguments
            simulation.set_input(variable, period, _input_value(variable, seed))
        elif kind == "calculate":
            variable, period = arguments
            return _outcome(lambda: simulation.calculate(variable, period))
        elif kind == "derivative":
            (period,) = arguments
            return _outcome(
                lambda: simulation.derivative("income_tax", "salary", period, delta=100)
            )
        elif kind == "reform":
            (rate,) = arguments
            simulation.apply_reform({"taxes.income_tax_rate": rate})
        elif kind == "delete_arrays":
            (variable,) = arguments
            simulation.delete_arrays(variable)
        elif kind == "new_holders":
            for population in simulation.populations.values():
                population._holders = {}
        elif kind == "drop" and member.parent is not None:
            self.drop(member)

    def drop(self, member: _Member) -> None:
        dropped = [member]
        for other in self.members:
            ancestor = other.parent
            while ancestor is not None and ancestor not in dropped:
                ancestor = ancestor.parent
            if ancestor is not None:
                dropped.append(other)
        self.members = [other for other in self.members if other not in dropped]
        if member.branch_name is not None:
            branches = member.parent.simulation.branches
            if branches.get(member.branch_name) is member.simulation:
                del branches[member.branch_name]
        del dropped, member
        gc.collect()

    def missing_files(self) -> List[str]:
        return [
            path
            for member in self.members
            for population in member.simulation.populations.values()
            for holder in population._holders.values()
            if holder._disk_storage is not None
            for path in holder._disk_storage._files.values()
            if not os.path.isfile(path)
        ]


def _outcome(calculate):
    try:
        return np.asarray(calculate()).tolist()
    except Exception as error:  # Both storages must fail the same way.
        return type(error).__name__


_index = st.integers(0, 15)
_operations = st.lists(
    st.one_of(
        st.tuples(st.just("branch"), _index, st.sampled_from(NAMES)),
        st.tuples(st.just("clone"), _index),
        st.tuples(
            st.just("set"),
            _index,
            st.sampled_from(INPUTS),
            st.sampled_from(MONTHS),
            st.integers(0, 5_000),
        ),
        st.tuples(
            st.just("calculate"),
            _index,
            st.sampled_from(CALCULATED),
            st.sampled_from(MONTHS),
        ),
        st.tuples(st.just("derivative"), _index, st.sampled_from(MONTHS)),
        st.tuples(st.just("reform"), _index, st.sampled_from([0.1, 0.2, 0.3])),
        st.tuples(
            st.just("delete_arrays"), _index, st.sampled_from(INPUTS + CALCULATED)
        ),
        st.tuples(st.just("new_holders"), _index),
        st.tuples(st.just("drop"), _index),
    ),
    max_size=30,
)


@hypothesis.settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.example(
    operations=[
        ("set", 0, "salary", "2017-01", 1000),
        ("clone", 0),
        ("new_holders", 0),
    ]
)
@hypothesis.example(
    operations=[
        ("set", 0, "salary", "2017-01", 1000),
        ("clone", 0),
        ("new_holders", 0),
        ("set", 0, "salary", "2017-01", 2000),
        ("calculate", 1, "salary", "2017-01"),
    ]
)
@hypothesis.given(operations=_operations)
def test_disk_storage_gives_the_results_memory_storage_gives(operations):
    """For any sequence of branching, cloning, inputs, calculations,
    derivatives, reforms and drops:

    - every calculation returns what it returns with values held in memory;
    - every file a live storage maps is on disk;
    - once the simulations are gone, their directories are removed, and no
      finalizer raised.
    """
    with unraisable_exceptions() as errors:
        in_memory = _Family(on_disk=False)
        on_disk = _Family(on_disk=True)
        for operation in operations:
            expected = in_memory.apply(operation)
            assert on_disk.apply(operation) == expected, operation
            assert on_disk.missing_files() == [], operation

        for member_in_memory, member_on_disk in zip(in_memory.members, on_disk.members):
            for variable in CALCULATED:
                for period in MONTHS:
                    assert _outcome(
                        lambda: member_on_disk.simulation.calculate(variable, period)
                    ) == _outcome(
                        lambda: member_in_memory.simulation.calculate(variable, period)
                    ), (variable, period)

        directories = {
            member.simulation._data_storage_dir
            for member in on_disk.members
            if member.simulation._data_storage_dir is not None
        }
        del in_memory, on_disk, member_in_memory, member_on_disk
        gc.collect()

    assert errors == []
    assert [directory for directory in directories if os.path.exists(directory)] == []
