"""Disk-backed holder storage keeps the values each simulation reads.

Every ``OnDiskStorage`` writes into a directory of its own and reads the files
it inherited from the storage it was cloned from. A directory stays on disk
while any storage may read a file in it, and a file a clone reads is never
overwritten. These tests pin that for branches, clones, ``derivative`` and
holders that a branch creates itself; ``test_disk_storage_differential.py``
checks it as a property over random sequences of operations.
"""

from __future__ import annotations

import copy
import gc
import os
import pickle
import shutil
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.enums import EnumArray
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.variables import Variable
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

    del clone
    gc.collect()
    storage.put(np.array([5.0]), PERIOD)

    assert len(os.listdir(storage.storage_dir)) == 2
    np.testing.assert_array_equal(storage.get(PERIOD), [5.0])


def test_reforms_after_a_derivative_reuse_the_files():
    """Once ``derivative``'s clone is gone, recalculating overwrites in place."""
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))
    simulation.calculate("income_tax", PERIOD)
    simulation.derivative("income_tax", "salary", PERIOD, delta=100)
    gc.collect()
    storage = simulation.get_holder("income_tax")._disk_storage

    for rate in (0.1, 0.2, 0.3, 0.4):
        simulation.apply_reform({"taxes.income_tax_rate": rate})
        np.testing.assert_allclose(
            simulation.calculate("income_tax", PERIOD), [3_000.0 * rate]
        )

    assert len(os.listdir(storage.storage_dir)) == 1


def test_a_live_branch_adds_at_most_one_file_per_key():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1.0]))
    branch = simulation.get_branch("branch")
    storage = simulation.get_holder("salary")._disk_storage

    for value in (2.0, 3.0, 4.0):
        simulation.delete_arrays("salary", PERIOD)
        simulation.set_input("salary", PERIOD, np.array([value]))

    assert len(os.listdir(storage.storage_dir)) == 2
    np.testing.assert_array_equal(branch.calculate("salary", PERIOD), [1.0])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [4.0])


def test_derivative_of_a_carried_over_input_leaves_it():
    """A year input carried over to later years is not perturbed by
    ``derivative`` (from the carry-over-order review's repro)."""

    class carried(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "Carried-over input"

    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.auto_carry_over_input_variables = True
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system, count=1
    )
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    # Its holder is created after memory_config, so it stores on disk.
    tax_benefit_system.add_variables(carried)
    simulation.set_input("carried", "2012", [100])

    derivative = simulation.derivative("carried", "carried", "2012", delta=1)

    np.testing.assert_array_equal(derivative, [1.0])
    np.testing.assert_array_equal(simulation.get_array("carried", "2012"), [100.0])
    np.testing.assert_array_equal(simulation.calculate("carried", "2013"), [100.0])


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


def test_storing_after_restore_leaves_the_restored_files(tmp_path):
    """Another storage that restored the same directory keeps its values."""
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    writer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    writer.put(np.array([1.0]), PERIOD)
    first = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    first.restore()
    second = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    second.restore()

    first.put(np.array([2.0]), PERIOD)

    np.testing.assert_array_equal(second.get(PERIOD), [1.0])
    np.testing.assert_array_equal(first.get(PERIOD), [2.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [2.0])


def test_a_write_after_restore_leaves_older_readers_and_is_restored_next(tmp_path):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    writer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    writer.put(np.array([1.0]), PERIOD)
    clone = writer.clone()
    writer.put(np.array([2.0]), PERIOD)
    restorer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    restorer.restore()

    restorer.put(np.array([3.0]), PERIOD)

    np.testing.assert_array_equal(clone.get(PERIOD), [1.0])
    np.testing.assert_array_equal(writer.get(PERIOD), [2.0])
    np.testing.assert_array_equal(restorer.get(PERIOD), [3.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [3.0])


def test_two_storages_writing_after_restore_keep_their_own_values(tmp_path):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    OnDiskStorage(str(storage_dir), preserve_storage_dir=True).put(
        np.array([1.0]), PERIOD
    )
    first = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    first.restore()
    second = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    second.restore()

    first.put(np.array([2.0]), PERIOD)
    second.put(np.array([3.0]), PERIOD)

    np.testing.assert_array_equal(first.get(PERIOD), [2.0])
    np.testing.assert_array_equal(second.get(PERIOD), [3.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [3.0])


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


def test_a_preserved_storage_in_the_simulation_directory_stays():
    simulation = build_simulation(on_disk=False)
    storage = simulation.get_holder("salary").create_disk_storage(preserve=True)
    storage.put(np.array([1.0]), PERIOD)
    data_storage_dir = simulation.data_storage_dir

    del simulation
    gc.collect()

    try:
        np.testing.assert_array_equal(storage.get(PERIOD), [1.0])
    finally:
        shutil.rmtree(data_storage_dir, ignore_errors=True)


def test_preserving_a_holder_storage_keeps_it_after_the_family_goes():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1.0]))
    storage = simulation.get_holder("salary")._disk_storage
    storage.preserve_storage_dir = True
    data_storage_dir, storage_dir = simulation.data_storage_dir, storage.storage_dir

    del simulation, storage
    gc.collect()

    try:
        assert os.listdir(storage_dir) == [f"default_{PERIOD}.npy"]
    finally:
        shutil.rmtree(data_storage_dir, ignore_errors=True)


@pytest.mark.parametrize("copy_storage", [copy.copy, copy.deepcopy, "pickle"])
def test_a_copy_of_a_storage_is_independent_of_it(tmp_path, copy_storage):
    """A copy reads the original's files, keeps them, and writes its own."""
    storage = OnDiskStorage.temporary("salary", str(tmp_path))
    storage.put(np.array([1.0]), PERIOD)
    if copy_storage == "pickle":
        copied = pickle.loads(pickle.dumps(storage))
    else:
        copied = copy_storage(storage)

    copied.put(np.array([2.0]), PERIOD)
    storage.put(np.array([3.0]), PERIOD)
    np.testing.assert_array_equal(copied.get(PERIOD), [2.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [3.0])

    storage_dir = storage.storage_dir
    del storage
    gc.collect()

    assert os.path.isdir(storage_dir)
    np.testing.assert_array_equal(copied.get(PERIOD), [2.0])


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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_and_its_parent_keep_their_own_values(tmp_path):
    storage = OnDiskStorage.temporary("salary", str(tmp_path))
    storage.put(np.array([1.0]), PERIOD)
    parent_wrote_read, parent_wrote_write = os.pipe()
    child_read_read, child_read_write = os.pipe()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            storage.put(np.array([2.0]), PERIOD)
            os.read(parent_wrote_read, 1)
            os.write(child_read_write, str(storage.get(PERIOD)[0]).encode())
        finally:
            os._exit(0)
    storage.put(np.array([3.0]), PERIOD)
    os.write(parent_wrote_write, b"x")
    child_value = float(os.read(child_read_read, 32).decode())
    os.waitpid(pid, 0)

    assert child_value == 2.0
    np.testing.assert_array_equal(storage.get(PERIOD), [3.0])
