"""Disk-backed holder storage keeps the values each simulation reads.

Each newly created transient holder gets its own directory. Cloned disk
storages share their source's directory and files, keep them alive, and write
new files whenever a clone still reads the old value. These tests pin that for
branches, clones, ``derivative`` and holders created after replacing a holder;
``test_disk_storage_differential.py`` checks it over random operation sequences.
"""

from __future__ import annotations

import copy
import gc
import os
import pickle
import shutil
import tempfile

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


def _storage(tmp_path, name: str = "salary") -> OnDiskStorage:
    return OnDiskStorage(tempfile.mkdtemp(prefix=f"{name}_", dir=tmp_path))


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
    """A clone writing the same key leaves its source's value readable."""
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
    # A branch creating a new holder gets a directory of its own.
    simulation.data_storage_dir
    branch = simulation.get_branch("branch")
    branch.set_input("salary", PERIOD, np.array([1.0]))
    del branch
    _drop_branch(simulation, "branch")

    simulation.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [2.0])


def test_directory_of_a_holder_created_by_a_branch_goes_with_the_branch():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")
    # A copied salary storage shares the source directory, while this new
    # holder uses the branch's own lazily created simulation directory.
    branch.calculate("income_tax", PERIOD)
    root_dir = simulation.get_holder("salary")._disk_storage.storage_dir
    branch_dir = branch.get_holder("income_tax")._disk_storage.storage_dir
    assert branch_dir != root_dir

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


@pytest.mark.parametrize("clone_first", [False, True], ids=["root", "clone"])
def test_a_data_storage_dir_set_by_the_user_stays(tmp_path, clone_first):
    directory = tmp_path / "simulation"
    simulation = build_simulation()
    simulation._data_storage_dir = str(directory)
    storer = simulation.clone() if clone_first else simulation
    assert not directory.exists()

    # A chosen folder, or a clone's chosen parent, may not exist at first use.
    data_storage_dir = storer.data_storage_dir
    assert directory.is_dir()
    assert os.path.isdir(data_storage_dir)
    if clone_first:
        assert os.path.dirname(data_storage_dir) == str(directory)
    else:
        assert data_storage_dir == str(directory)
    storer.set_input("salary", PERIOD, np.array([1.0]))
    storage_dir = storer.get_holder("salary")._disk_storage.storage_dir
    assert os.path.dirname(storage_dir) == data_storage_dir
    np.testing.assert_array_equal(storer.calculate("salary", PERIOD), [1.0])

    del storer, simulation
    gc.collect()

    assert not os.path.exists(storage_dir)
    assert directory.is_dir()


@pytest.mark.parametrize("preserve", [False, True])
def test_explicit_holder_storage_keeps_the_variable_directory_layout(
    tmp_path, preserve
):
    simulation = build_simulation(on_disk=False)
    directory = tmp_path / "dump"
    storage = simulation.get_holder("salary").create_disk_storage(
        directory=str(directory), preserve=preserve
    )
    storage.put(np.array([1.0]), PERIOD)

    assert storage.storage_dir == str(directory / "salary")
    np.testing.assert_array_equal(
        np.load(directory / "salary" / f"default_{PERIOD}.npy"), [1.0]
    )


def test_preserved_holder_storage_keeps_the_variable_directory_layout():
    simulation = build_simulation(on_disk=False)
    storage = simulation.get_holder("salary").create_disk_storage(preserve=True)
    storage.put(np.array([1.0]), PERIOD)
    directory = simulation.data_storage_dir

    try:
        assert storage.storage_dir == os.path.join(directory, "salary")
        np.testing.assert_array_equal(
            np.load(os.path.join(directory, "salary", f"default_{PERIOD}.npy")),
            [1.0],
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_a_key_stored_again_overwrites_its_file_until_a_clone_reads_it(tmp_path):
    storage = _storage(tmp_path)
    storage.put(np.array([1.0]), PERIOD)
    first_path = storage._files[f"default_{PERIOD}"]
    storage.put(np.array([2.0]), PERIOD)
    assert storage._files[f"default_{PERIOD}"] == first_path

    clone = storage.clone()
    storage.put(np.array([3.0]), PERIOD)
    replacement_path = storage._files[f"default_{PERIOD}"]
    storage.put(np.array([4.0]), PERIOD)

    assert replacement_path != first_path
    assert storage._files[f"default_{PERIOD}"] == replacement_path
    np.testing.assert_array_equal(clone.get(PERIOD), [2.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [4.0])

    del clone
    gc.collect()
    storage.put(np.array([5.0]), PERIOD)

    assert storage._files[f"default_{PERIOD}"] == replacement_path
    np.testing.assert_array_equal(storage.get(PERIOD), [5.0])


def test_reforms_after_a_derivative_reuse_the_files():
    """Repeated reforms reuse the storage's first unshared replacement."""
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))
    simulation.calculate("income_tax", PERIOD)
    simulation.derivative("income_tax", "salary", PERIOD, delta=100)
    gc.collect()
    storage = simulation.get_holder("income_tax")._disk_storage

    replacement_path = None
    for rate in (0.1, 0.2, 0.3, 0.4):
        simulation.apply_reform({"taxes.income_tax_rate": rate})
        np.testing.assert_allclose(
            simulation.calculate("income_tax", PERIOD), [3_000.0 * rate]
        )
        path = storage._files[f"default_{PERIOD}"]
        if replacement_path is None:
            replacement_path = path
        else:
            assert path == replacement_path


def test_a_live_branch_adds_at_most_one_file_per_key():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1.0]))
    branch = simulation.get_branch("branch")
    storage = simulation.get_holder("salary")._disk_storage
    source_path = storage._files[f"default_{PERIOD}"]

    replacement_paths = set()
    for value in (2.0, 3.0, 4.0):
        simulation.delete_arrays("salary", PERIOD)
        simulation.set_input("salary", PERIOD, np.array([value]))
        replacement_paths.add(storage._files[f"default_{PERIOD}"])

    assert len(replacement_paths) == 1
    assert source_path not in replacement_paths
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
    storage = _storage(tmp_path)
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


def test_restore_reads_direct_key_files_after_a_clone_shares_them(tmp_path):
    """Restore excludes replacements, as documented by ``OnDiskStorage``."""
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
    np.testing.assert_array_equal(restored.get(PERIOD), [1.0])
    np.testing.assert_array_equal(restored.get(PERIOD, "a.1"), [5.0])
    np.testing.assert_array_equal(restored.get("2017-02"), [3.0])
    np.testing.assert_array_equal(clone.get(PERIOD), [1.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [2.0])


def test_storing_after_restore_updates_the_direct_key_file(tmp_path):
    """Without cloning, restore continues writing the direct key file."""
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    writer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    writer.put(np.array([1.0]), PERIOD)
    first = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    first.restore()

    first.put(np.array([2.0]), PERIOD)

    np.testing.assert_array_equal(first.get(PERIOD), [2.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [2.0])


def test_a_write_after_restore_leaves_the_restored_storage_clone(tmp_path):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    writer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    writer.put(np.array([1.0]), PERIOD)
    restorer = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    restorer.restore()
    clone = restorer.clone()

    restorer.put(np.array([3.0]), PERIOD)

    np.testing.assert_array_equal(clone.get(PERIOD), [1.0])
    np.testing.assert_array_equal(writer.get(PERIOD), [1.0])
    np.testing.assert_array_equal(restorer.get(PERIOD), [3.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [1.0])


def test_two_restored_storage_families_keep_their_own_values(tmp_path):
    storage_dir = tmp_path / "salary"
    storage_dir.mkdir()
    OnDiskStorage(str(storage_dir), preserve_storage_dir=True).put(
        np.array([1.0]), PERIOD
    )
    first = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    first.restore()
    second = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    second.restore()
    # Each family protects the file once a clone may read it. Independently
    # restored storages alone have no cross-family snapshot guarantee.
    first_clone = first.clone()
    second_clone = second.clone()

    first.put(np.array([2.0]), PERIOD)
    second.put(np.array([3.0]), PERIOD)

    np.testing.assert_array_equal(first.get(PERIOD), [2.0])
    np.testing.assert_array_equal(second.get(PERIOD), [3.0])
    np.testing.assert_array_equal(first_clone.get(PERIOD), [1.0])
    np.testing.assert_array_equal(second_clone.get(PERIOD), [1.0])
    latest = OnDiskStorage(str(storage_dir), preserve_storage_dir=True)
    latest.restore()
    np.testing.assert_array_equal(latest.get(PERIOD), [1.0])


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
    path = storage._files[f"default_{PERIOD}"]

    del simulation, storage
    gc.collect()

    try:
        assert os.path.isdir(storage_dir)
        np.testing.assert_array_equal(np.load(path), [1.0])
    finally:
        shutil.rmtree(data_storage_dir, ignore_errors=True)


@pytest.mark.parametrize("copy_storage", [copy.copy, copy.deepcopy, "pickle"])
def test_a_copy_of_a_storage_is_independent_of_it(tmp_path, copy_storage):
    """A copy reads the original's files, keeps them, and writes its own."""
    storage = _storage(tmp_path)
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
    storage = _storage(tmp_path, "status")
    status = CountryTaxBenefitSystem().variables["housing_occupancy_status"]
    storage.put(status.possible_values.encode(np.array(["owner"])), PERIOD)
    assert isinstance(storage.get(PERIOD), EnumArray)

    storage.put(np.array([3]), PERIOD)

    assert not isinstance(storage.get(PERIOD), EnumArray)
