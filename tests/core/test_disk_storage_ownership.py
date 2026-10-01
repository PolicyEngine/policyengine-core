"""Disk-backed holder storage keeps the values each simulation reads.

Every ``OnDiskStorage`` writes into a directory of its own and reads the files
it inherited from the storage it was cloned from. A directory stays on disk
while any storage may read a file in it, and a file a clone reads is never
overwritten. These tests pin that for branches, clones, ``derivative`` and
holders that a branch creates itself; ``test_disk_storage_differential.py``
checks it as a property over random sequences of operations.
"""

from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.enums import EnumArray
from policyengine_core.simulations import Simulation
from tests.fixtures.disk_storage import build_simulation, unraisable_exceptions

PERIOD = "2017-01"


def _drop_branch(simulation: Simulation, name: str) -> None:
    del simulation.branches[name]
    gc.collect()


def test_branch_holder_leaves_the_parent_files():
    """A holder a branch creates keeps its hands off the parent's files."""
    simulation = build_simulation()
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
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    clone = simulation.clone()

    clone.set_input("salary", PERIOD, np.array([5.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1_000.0])
    np.testing.assert_array_equal(clone.calculate("salary", PERIOD), [5.0])


@pytest.mark.parametrize("on_disk", [False, True])
def test_derivative_leaves_the_simulation_inputs(on_disk):
    """``derivative`` perturbs a clone's input, not the simulation's own."""
    simulation = build_simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))

    derivative = simulation.derivative("income_tax", "salary", PERIOD, delta=100)

    np.testing.assert_allclose(derivative, [0.15])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [3_000.0])


def test_branches_with_the_same_name_keep_their_own_values():
    simulation = build_simulation()
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
    simulation = build_simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")

    simulation.delete_arrays("salary", PERIOD)
    simulation.set_input("salary", PERIOD, np.array([7.0]))

    np.testing.assert_array_equal(branch.calculate("salary", PERIOD), [1_000.0])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [7.0])


def test_simulation_stores_on_disk_after_its_branch_storage_is_removed():
    simulation = build_simulation()
    # Branches made once the directory exists share it.
    simulation.data_storage_dir
    branch = simulation.get_branch("branch")
    branch.set_input("salary", PERIOD, np.array([1.0]))
    del branch
    _drop_branch(simulation, "branch")

    simulation.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [2.0])


def test_branch_directory_goes_with_the_branch():
    simulation = build_simulation()
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
    with unraisable_exceptions() as errors:
        simulation = build_simulation()
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
    simulation = build_simulation()
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
