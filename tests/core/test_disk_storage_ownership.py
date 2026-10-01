"""Disk-backed holder storage keeps the values each simulation reads.

Every ``OnDiskStorage`` writes into a directory of its own and reads the files
it inherited from the storage it was cloned from. A directory stays on disk
while any storage may read a file in it, and a file a clone reads is never
overwritten. These tests pin that for branches, clones and holders that a
branch creates itself, and check, as a property over random sequences of
operations, that storing values on disk gives the same results as storing
them in memory.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.enums import EnumArray
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder

PERIOD = "2017-01"


def _simulation(on_disk: bool = True, count: int = 1) -> Simulation:
    # A tax-benefit system keeps the last simulation built on it, so give
    # each simulation its own for the tests that release one.
    simulation = SimulationBuilder().build_default_simulation(
        CountryTaxBenefitSystem(), count=count
    )
    if on_disk:
        simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    # Holders made before ``memory_config`` is set have no disk storage, so
    # start without any: each holder is created where it is first used.
    for population in simulation.populations.values():
        population._holders = {}
    return simulation


@contextlib.contextmanager
def _unraisable_exceptions():
    """Collect exceptions Python can only print, such as those in finalizers."""
    errors: List[str] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda unraisable: errors.append(repr(unraisable.exc_value))
    try:
        yield errors
    finally:
        sys.unraisablehook = previous_hook


def _drop_branch(simulation: Simulation, name: str) -> None:
    del simulation.branches[name]
    gc.collect()


def test_branch_holder_leaves_the_parent_files():
    """A holder a branch creates keeps its hands off the parent's files."""
    simulation = _simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")
    # The branch creates its own income_tax holder, then the root creates one.
    branch.calculate("income_tax", PERIOD)
    expected = simulation.calculate("income_tax", PERIOD).copy()

    del branch
    _drop_branch(simulation, "branch")

    np.testing.assert_array_equal(
        simulation.get_holder("income_tax").get_array(PERIOD), expected
    )


def test_clone_input_leaves_the_source_input():
    """``Simulation.clone`` keeps the branch name, but not the files."""
    simulation = _simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    clone = simulation.clone()

    clone.set_input("salary", PERIOD, np.array([5.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1_000.0])
    np.testing.assert_array_equal(clone.calculate("salary", PERIOD), [5.0])


@pytest.mark.parametrize("on_disk", [False, True])
def test_derivative_leaves_the_simulation_inputs(on_disk):
    """``derivative`` perturbs a clone's input, not the simulation's own."""
    simulation = _simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))

    derivative = simulation.derivative("income_tax", "salary", PERIOD, delta=100)

    np.testing.assert_allclose(derivative, [0.15])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [3_000.0])


def test_branches_with_the_same_name_keep_their_own_values():
    simulation = _simulation()
    # Branches made after the root has stored on disk share its directory.
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    first = simulation.get_branch("first").get_branch("nested")
    second = simulation.get_branch("second").get_branch("nested")

    first.set_input("salary", PERIOD, np.array([1.0]))
    second.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(first.calculate("salary", PERIOD), [1.0])
    np.testing.assert_array_equal(second.calculate("salary", PERIOD), [2.0])


@pytest.mark.parametrize("on_disk", [False, True])
def test_branch_keeps_the_values_it_started_with(on_disk):
    """A parent replacing a value after branching leaves the branch's value."""
    simulation = _simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")

    simulation.delete_arrays("salary", PERIOD)
    simulation.set_input("salary", PERIOD, np.array([7.0]))

    np.testing.assert_array_equal(branch.calculate("salary", PERIOD), [1_000.0])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [7.0])


def test_simulation_stores_on_disk_after_its_branch_storage_is_removed():
    simulation = _simulation()
    # Branches made once the directory exists share it.
    simulation.data_storage_dir
    branch = simulation.get_branch("branch")
    branch.set_input("salary", PERIOD, np.array([1.0]))
    del branch
    _drop_branch(simulation, "branch")

    simulation.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [2.0])


def test_branch_directory_goes_with_the_branch():
    simulation = _simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")
    branch.set_input("salary", PERIOD, np.array([2.0]))
    root_dir = simulation.get_holder("salary")._disk_storage.storage_dir
    branch_dir = branch.get_holder("salary")._disk_storage.storage_dir
    assert branch_dir != root_dir
    assert os.path.dirname(branch_dir) == simulation.data_storage_dir

    del branch
    _drop_branch(simulation, "branch")

    assert not os.path.exists(branch_dir)
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1_000.0])


def test_releasing_a_family_removes_its_directory_without_errors():
    with _unraisable_exceptions() as errors:
        simulation = _simulation()
        simulation.set_input("salary", PERIOD, np.array([1_000.0]))
        branch = simulation.get_branch("branch")
        branch.calculate("income_tax", PERIOD)
        simulation.calculate("income_tax", PERIOD)
        directory = simulation.data_storage_dir
        assert os.path.isdir(directory)

        del branch
        _drop_branch(simulation, "branch")
        del simulation
        gc.collect()

    assert errors == []
    assert not os.path.exists(directory)


def test_a_data_storage_dir_set_by_the_user_stays(tmp_path):
    directory = tmp_path / "simulation"
    simulation = _simulation()
    simulation._data_storage_dir = str(directory)
    simulation.set_input("salary", PERIOD, np.array([1.0]))
    storage_dir = simulation.get_holder("salary")._disk_storage.storage_dir
    assert os.path.dirname(storage_dir) == str(directory)

    del simulation
    gc.collect()

    assert not os.path.exists(storage_dir)
    assert directory.is_dir()


def test_a_key_stored_again_overwrites_its_file_until_a_clone_reads_it(tmp_path):
    storage = OnDiskStorage.temporary("salary", str(tmp_path))
    storage.put(np.array([1.0]), PERIOD)
    storage.put(np.array([2.0]), PERIOD)
    assert len(os.listdir(storage.storage_dir)) == 1

    clone = storage.clone()
    storage.put(np.array([3.0]), PERIOD)
    storage.put(np.array([4.0]), PERIOD)

    assert len(os.listdir(storage.storage_dir)) == 2
    np.testing.assert_array_equal(clone.get(PERIOD), [2.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [4.0])


def test_a_clone_keeps_the_directories_it_reads(tmp_path):
    storage = OnDiskStorage.temporary("salary", str(tmp_path))
    storage.put(np.array([1.0]), PERIOD)
    storage_dir = storage.storage_dir
    grandchild = storage.clone().clone()

    del storage
    gc.collect()

    assert os.path.isdir(storage_dir)
    np.testing.assert_array_equal(grandchild.get(PERIOD), [1.0])

    del grandchild
    gc.collect()

    assert not os.path.exists(storage_dir)


def test_restore_reads_the_latest_file_of_each_key(tmp_path):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    storage = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    storage.put(np.array([1.0]), PERIOD)
    storage.put(np.array([5.0]), PERIOD, "a.1")
    clone = storage.clone()
    storage.put(np.array([2.0]), PERIOD)
    storage.put(np.array([3.0]), "2017-02")

    restored = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    restored.restore()

    assert sorted(restored._files) == [
        "a.1_2017-01",
        "default_2017-01",
        "default_2017-02",
    ]
    np.testing.assert_array_equal(restored.get(PERIOD), [2.0])
    np.testing.assert_array_equal(restored.get(PERIOD, "a.1"), [5.0])
    np.testing.assert_array_equal(restored.get("2017-02"), [3.0])
    np.testing.assert_array_equal(clone.get(PERIOD), [1.0])


def test_storage_removes_only_its_own_directory(tmp_path):
    storage_dir = tmp_path / "parent" / "salary"
    storage_dir.mkdir(parents=True)
    storage = OnDiskStorage(str(storage_dir))
    storage.put(np.array([1.0]), PERIOD)

    del storage
    gc.collect()

    assert not storage_dir.exists()
    assert (tmp_path / "parent").is_dir()


@pytest.mark.parametrize("preserve_when", ["created", "later"])
def test_preserved_storage_directory_stays(tmp_path, preserve_when):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    storage = OnDiskStorage(
        str(storage_dir), preserve_storage_dir=preserve_when == "created"
    )
    storage.preserve_storage_dir = True
    storage.put(np.array([1.0]), PERIOD)

    del storage
    gc.collect()

    assert (storage_dir / f"default_{PERIOD}.npy").is_file()


def test_a_plain_array_stored_over_an_enum_reads_back_plain(tmp_path):
    storage = OnDiskStorage.temporary("status", str(tmp_path))
    status = CountryTaxBenefitSystem().variables["housing_occupancy_status"]
    storage.put(status.possible_values.encode(np.array(["owner"])), PERIOD)
    assert isinstance(storage.get(PERIOD), EnumArray)

    storage.put(np.array([3]), PERIOD)

    assert not isinstance(storage.get(PERIOD), EnumArray)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_leaves_the_parent_directories(tmp_path):
    storage = OnDiskStorage.temporary("salary", str(tmp_path))
    storage.put(np.array([1.0]), PERIOD)
    storage_dir = storage.storage_dir

    with warnings.catch_warnings():
        # Python warns that forking a multi-threaded process may deadlock.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            del storage
            gc.collect()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)

    assert os.path.isdir(storage_dir)
    np.testing.assert_array_equal(storage.get(PERIOD), [1.0])


# Differential property: on disk as in memory.

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
        self.members = [_Member(_simulation(on_disk, count=COUNT), None, None)]

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


@settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations=_operations)
def test_disk_storage_gives_the_results_memory_storage_gives(operations):
    """For any sequence of branching, cloning, inputs, calculations,
    derivatives, reforms and drops:

    - every calculation returns what it returns with values held in memory;
    - every file a live storage maps is on disk;
    - once the simulations are gone, their directory is removed, and no
      finalizer raised.
    """
    with _unraisable_exceptions() as errors:
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

        directory = on_disk.members[0].simulation._data_storage_dir
        del in_memory, on_disk, member_in_memory, member_on_disk
        gc.collect()

    assert errors == []
    assert directory is None or not os.path.exists(directory)
