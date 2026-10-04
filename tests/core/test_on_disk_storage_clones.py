"""Storages cloned from one another never write over each other's files.

A cloned ``OnDiskStorage`` shares its source's directory and the files stored
before cloning, and the two can each store the same key later, which names
the same path. ``put`` wrote every value to its key's path, so a write
through one storage changed what the other read:

* ``Simulation.clone()`` keeps the branch name, so a ``set_input`` on the
  clone wrote over the source's file for that period. A yearly input given
  through a ``set_input`` helper, which now replaces a month the simulation
  calculated, did the same.
* The source writing a period again after cloning (say, calculating it after
  deleting it) changed the clone's value too.

``put`` now writes over one of the family's files only if the storage wrote
it and has not shared it since; otherwise it writes a new file of its own,
in a subdirectory ``restore`` does not read. Storages that were not cloned
from one another write over each other's files, as before.
"""

from __future__ import annotations

import copy
import glob
import os
import pickle

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.experimental import MemoryConfig
from tests.fixtures.uprating_order import build_system, simulation

SYSTEMS = {True: build_system(True), False: build_system(False)}
HELPER_INPUTS = {
    "uprated_monthly_stock": [7.0, 9.0],
    "uprated_monthly": [84.0, 108.0],
}


@pytest.fixture(params=[True, False], ids=["carry_over", "no_carry_over"])
def system(request):
    return SYSTEMS[request.param]


def _on_disk(built, variable):
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder(variable)
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    return holder


def _stored(holder, period):
    """The stored value, read past the simulation's fast cache."""
    return holder.get_array(periods.period(period))


@pytest.mark.parametrize("variable", list(HELPER_INPUTS), ids=["dispatch", "divide"])
def test_helper_input_on_a_clone_leaves_the_source_unchanged(system, variable):
    parent = simulation(system, {})
    holder = _on_disk(parent, variable)
    calculated = parent.calculate(variable, "2012-12").copy()
    assert holder._disk_storage.is_derived(periods.period("2012-12"))

    child = parent.clone()
    child.set_input(variable, "2012", np.array(HELPER_INPUTS[variable]))

    np.testing.assert_array_equal(child.calculate(variable, "2012-12"), [7, 9])
    np.testing.assert_array_equal(_stored(holder, "2012-12"), calculated)
    assert holder.is_derived(periods.period("2012-12"))


def test_set_input_on_a_clone_leaves_the_source_unchanged():
    parent = simulation(SYSTEMS[True], {})
    holder = _on_disk(parent, "uprated")
    parent.set_input("uprated", "2012", np.array([1.0, 2.0]))

    child = parent.clone()
    child.set_input("uprated", "2012", np.array([3.0, 4.0]))

    np.testing.assert_array_equal(_stored(holder, "2012"), [1, 2])
    np.testing.assert_array_equal(_stored(child.get_holder("uprated"), "2012"), [3, 4])


def test_source_writing_a_period_again_leaves_the_clone_unchanged():
    parent = simulation(SYSTEMS[True], {})
    _on_disk(parent, "uprated")
    parent.set_input("uprated", "2011", np.array([1001.0, 77.0]))
    parent.set_input("uprated", "2012", np.array([2222.0, 222.0]))
    branch = parent.get_branch("branch")
    expected = branch.calculate("uprated", "2015").copy()

    parent.delete_arrays("uprated", "2012")
    parent.calculate("uprated", "2012")

    np.testing.assert_array_equal(
        _stored(branch.get_holder("uprated"), "2012"), [2222, 222]
    )
    branch.delete_arrays("uprated", "2015")
    np.testing.assert_array_equal(branch.calculate("uprated", "2015"), expected)


# ----- Storage level ----- #


@pytest.fixture
def storage(tmp_path):
    return OnDiskStorage(str(tmp_path), preserve_storage_dir=True)


def _npy_files(storage):
    return sorted(
        glob.glob(os.path.join(storage.storage_dir, "**", "*.npy"), recursive=True)
    )


def test_storages_cloned_from_one_another_keep_their_own_values(storage):
    storage.put(np.array([1.0]), "2012")
    clone = storage.clone()

    storage.put(np.array([2.0]), "2012")
    np.testing.assert_array_equal(clone.get("2012"), [1])
    clone.put(np.array([3.0]), "2012")
    np.testing.assert_array_equal(storage.get("2012"), [2])
    np.testing.assert_array_equal(clone.get("2012"), [3])

    # A key both store after cloning.
    storage.put(np.array([4.0]), "2013")
    clone.put(np.array([5.0]), "2013")
    np.testing.assert_array_equal(storage.get("2013"), [4])
    np.testing.assert_array_equal(clone.get("2013"), [5])


def test_a_clone_of_a_clone_keeps_its_own_values(storage):
    storage.put(np.array([1.0]), "2012")
    clone = storage.clone()
    clone.put(np.array([2.0]), "2012")
    grandchild = clone.clone()

    clone.put(np.array([3.0]), "2012")
    grandchild.put(np.array([4.0]), "2012")
    storage.put(np.array([5.0]), "2012")

    np.testing.assert_array_equal(storage.get("2012"), [5])
    np.testing.assert_array_equal(clone.get("2012"), [3])
    np.testing.assert_array_equal(grandchild.get("2012"), [4])


def test_a_storage_writes_over_its_own_files(storage):
    """No file per write: a value the storage alone reads is replaced in
    place, including one it moved to a new file of its own."""
    storage.put(np.array([1.0]), "2012")
    storage.put(np.array([2.0]), "2012", derived=True)
    assert len(_npy_files(storage)) == 1
    np.testing.assert_array_equal(storage.get("2012"), [2])
    assert storage.is_derived(periods.period("2012"))

    storage.clone()
    storage.put(np.array([3.0]), "2012")
    storage.put(np.array([4.0]), "2012")
    assert len(_npy_files(storage)) == 2
    np.testing.assert_array_equal(storage.get("2012"), [4])
    assert not storage.is_derived(periods.period("2012"))


def test_storages_not_cloned_from_one_another_write_over_each_others_files(
    storage,
):
    """Two storages made for one directory, as ``restore`` and a separate
    writer are, still name the same file for a key."""
    storage.put(np.array([1.0]), "2012")
    writer = OnDiskStorage(storage.storage_dir, preserve_storage_dir=True)
    writer.put(np.array([2.0]), "2012")
    np.testing.assert_array_equal(storage.get("2012"), [2])
    assert len(_npy_files(storage)) == 1


def test_restore_reads_only_the_files_named_for_their_keys(storage):
    storage.put(np.array([1.0]), "2012")
    storage.put(np.array([2.0]), "2012-03")
    clone = storage.clone()
    clone.put(np.array([3.0]), "2012")
    assert len(_npy_files(storage)) == 3

    restored = OnDiskStorage(storage.storage_dir, preserve_storage_dir=True)
    restored.restore()

    assert sorted(map(str, restored.get_known_periods())) == [
        "2012",
        "2012-03",
    ]
    np.testing.assert_array_equal(restored.get("2012"), [1])


def test_a_restored_storage_and_its_clone_keep_their_own_values(storage):
    storage.put(np.array([1.0]), "2012")
    restored = OnDiskStorage(storage.storage_dir, preserve_storage_dir=True)
    restored.restore()
    clone = restored.clone()

    clone.put(np.array([2.0]), "2012")

    np.testing.assert_array_equal(restored.get("2012"), [1])
    np.testing.assert_array_equal(clone.get("2012"), [2])


@pytest.mark.parametrize(
    "copier", [copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))]
)
def test_a_copied_storage_and_its_source_keep_their_own_values(storage, copier):
    storage.put(np.array([1.0]), "2012")
    copied = copier(storage)

    copied.put(np.array([2.0]), "2012")
    np.testing.assert_array_equal(storage.get("2012"), [1])
    storage.put(np.array([3.0]), "2012")
    np.testing.assert_array_equal(copied.get("2012"), [2])
    np.testing.assert_array_equal(storage.get("2012"), [3])


def test_a_storage_unpickled_from_before_writes_over_none_of_its_files(storage):
    """A storage pickled before storages recorded the files they wrote may
    share them with another, so it writes a new file for each."""
    storage.put(np.array([1.0]), "2012")
    state = pickle.loads(pickle.dumps(storage.__dict__))
    del state["_own_paths"], state["_family_files"]
    old = OnDiskStorage.__new__(OnDiskStorage)
    old.__setstate__(state)

    old.put(np.array([2.0]), "2012")

    np.testing.assert_array_equal(storage.get("2012"), [1])
    np.testing.assert_array_equal(old.get("2012"), [2])


def test_a_storage_restored_then_written_still_writes_its_own_file(storage):
    """``restore`` shares nothing, so a storage that was never cloned keeps
    writing over the files it wrote, which another storage reads."""
    storage.put(np.array([5.0]), "2013")
    storage.restore()
    storage.put(np.array([6.0]), "2013")
    reader = OnDiskStorage(storage.storage_dir, preserve_storage_dir=True)
    reader.restore()
    np.testing.assert_array_equal(reader.get("2013"), [6])
    assert len(_npy_files(storage)) == 1


def test_deleting_and_writing_a_key_again_reuses_its_file(storage):
    storage.put(np.array([0.0]), "2012", derived=True)
    clone = storage.clone()
    for value in range(1, 21):
        storage.delete("2012")
        storage.put(np.array([float(value)]), "2012", derived=True)
    np.testing.assert_array_equal(storage.get("2012"), [20])
    np.testing.assert_array_equal(clone.get("2012"), [0])
    assert len(_npy_files(storage)) == 2
