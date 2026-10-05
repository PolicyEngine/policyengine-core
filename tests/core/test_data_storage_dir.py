"""The folder a simulation stores values on disk in is removed with it.

``Simulation.data_storage_dir`` made a temporary folder (``openfisca_*``) the
first time a simulation stored a value on disk (with a ``MemoryConfig``), and
nothing removed it reliably. Each disk storage removed its own subfolder
when collected, and the folder once empty, but not those still alive at
interpreter exit, so folders piled up. And the folder was shared in ways that
removed or replaced files something still read:

* A clone (or branch) made its new disk storages in its source's folder: for
  a variable both stored after cloning, the two wrote the same files, and
  whichever was collected first removed the other's.
* A pickled or deep-copied disk storage removed its subfolder when
  collected, while the storage it was copied from still read it, in this
  process or in the one that pickled it.
* A process forked from the simulation's did the same when it collected its
  copy of the simulation.
* The last disk storage collected removed the folder containing it once
  empty, even a folder the caller chose (``_data_storage_dir``).

Now a ``TemporaryStorageDirectory`` owns the folder a simulation makes, and
the simulation and every disk storage in the folder keep it alive; it is
removed when the last is collected or at interpreter exit, by the process
that made it. A clone makes its own folder for what it stores. The
properties over any sequence of these operations are in
``test_data_storage_dir_property.py``.
"""

from __future__ import annotations

import copy
import gc
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.data_storage.storage_directory import (
    TemporaryStorageDirectory,
)
from policyengine_core.enums import EnumArray
from tests.fixtures.data_storage_dir import (
    PEOPLE,
    VARIABLES,
    Level,
    disk_simulation,
    read,
    storage_folders,
    values,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fill(simulation, seed=0):
    """Store a value of every variable for 2015 and return them."""
    stored = {}
    for offset, variable in enumerate(VARIABLES):
        stored[variable] = values(variable, seed + offset)
        simulation.set_input(variable, "2015", stored[variable])
    return stored


def _assert_reads(simulation, stored, period="2015"):
    for variable, expected in stored.items():
        np.testing.assert_array_equal(
            np.asarray(read(simulation, variable, period)), expected
        )


def _files(folder):
    """Every file in ``folder``, as a path relative to it with ``/``."""
    return sorted(
        path.relative_to(folder).as_posix()
        for path in Path(folder).rglob("*")
        if path.is_file()
    )


def _run(script, tmp_path, **environment):
    """Run ``script`` in a new interpreter that makes temporary folders in
    ``tmp_path``, and return what it printed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env={**os.environ, "TMPDIR": str(tmp_path), **environment},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ----- No folder left behind ----- #


def test_a_simulation_storing_on_disk_leaves_no_folder_once_collected():
    simulation = disk_simulation()
    _fill(simulation)
    folder = simulation.data_storage_dir
    assert _files(folder) == [
        "disk_amount/default_2015.npy",
        "disk_level/default_2015.npy",
    ]

    del simulation
    gc.collect()

    assert not os.path.exists(folder)


def test_a_simulation_alive_at_exit_leaves_no_folder(tmp_path):
    # The usual leak: a country package's system keeps the last simulation
    # built with it alive until the interpreter exits.
    printed = _run(
        """
        from tests.fixtures.data_storage_dir import SYSTEM, disk_simulation, values

        simulation = disk_simulation()
        SYSTEM.simulation = simulation
        simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
        print(simulation.data_storage_dir)
        """,
        tmp_path,
    )
    folder = printed.split()[-1]
    assert Path(folder).parent.resolve() == tmp_path.resolve()
    assert not os.path.exists(folder)
    assert storage_folders(tmp_path) == []


def test_a_simulation_that_never_stores_on_disk_makes_no_folder():
    simulation = disk_simulation()
    simulation.memory_config = None
    simulation.set_input("disk_amount", "2015", values("disk_amount", 0))

    assert simulation._data_storage_dir is None


# ----- Nothing a live simulation or storage reads is removed ----- #


def test_a_clone_outliving_its_source_reads_every_value_on_disk():
    source = disk_simulation()
    stored = _fill(source)
    folder = source.data_storage_dir
    clone = source.clone()

    del source
    gc.collect()

    assert os.path.isdir(folder)
    _assert_reads(clone, stored)
    del clone
    gc.collect()
    assert not os.path.exists(folder)


def test_a_clone_of_a_branch_outliving_both_reads_every_value_on_disk():
    source = disk_simulation()
    stored = _fill(source)
    branch = source.get_branch("measurement")
    branch_stored = {"disk_amount": values("disk_amount", 7)}
    branch.set_input("disk_amount", "2016", branch_stored["disk_amount"])
    clone = branch.clone()
    folder = source.data_storage_dir
    # The branch stored its value through the disk storage it copied, in the
    # source's folder: it made none of its own.
    assert branch._data_storage_dir is None

    del source, branch
    gc.collect()

    _assert_reads(clone, stored)
    _assert_reads(clone, branch_stored, period="2016")
    del clone
    gc.collect()
    assert not os.path.exists(folder)


def test_a_clone_and_its_source_each_store_a_new_variable_in_their_own_folder():
    source = disk_simulation()
    # The source has made its folder before cloning.
    source.set_input("disk_level", "2015", values("disk_level", 0))
    clone = source.clone()
    source_amount = values("disk_amount", 1)
    clone_amount = values("disk_amount", 2)
    # Both store the same key, ``default_2015``, for a variable neither had.
    source.set_input("disk_amount", "2015", source_amount)
    clone.set_input("disk_amount", "2015", clone_amount)

    assert clone.data_storage_dir != source.data_storage_dir
    _assert_reads(source, {"disk_amount": source_amount})
    _assert_reads(clone, {"disk_amount": clone_amount})
    del clone
    gc.collect()
    _assert_reads(source, {"disk_amount": source_amount})


def test_a_holder_first_made_in_a_dropped_branch_leaves_the_parent_values():
    parent = disk_simulation()
    parent.set_input("disk_level", "2015", values("disk_level", 0))
    branch = parent.get_branch("inner")
    # The branch makes the variable's holder before its parent does.
    branch.set_input("disk_amount", "2015", values("disk_amount", 3))
    parent_amount = values("disk_amount", 4)
    parent.set_input("disk_amount", "2015", parent_amount)

    del parent.branches["inner"], branch
    gc.collect()

    _assert_reads(parent, {"disk_amount": parent_amount})


@pytest.mark.parametrize(
    "copier",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_a_copied_disk_storage_reads_its_values_after_the_simulation_is_collected(
    copier,
):
    simulation = disk_simulation()
    stored = _fill(simulation)
    folder = simulation.data_storage_dir
    copies = {
        variable: copier(simulation.persons._holders[variable]._disk_storage)
        for variable in VARIABLES
    }

    del simulation
    gc.collect()

    for variable, expected in stored.items():
        np.testing.assert_array_equal(
            np.asarray(copies[variable].get("2015")), expected
        )
    level = copies["disk_level"].get("2015")
    assert isinstance(level, EnumArray) and level.possible_values is Level
    del copies
    gc.collect()
    assert not os.path.exists(folder)


@pytest.mark.parametrize(
    "copier",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_a_copied_disk_storage_collected_first_leaves_the_simulation_values(copier):
    simulation = disk_simulation()
    stored = _fill(simulation)
    copied = copier(simulation.persons._holders["disk_amount"]._disk_storage)

    assert copied.preserve_storage_dir
    del copied
    gc.collect()

    _assert_reads(simulation, stored)


def test_a_disk_storage_unpickled_in_another_process_leaves_the_folder(tmp_path):
    simulation = disk_simulation()
    stored = _fill(simulation)
    folder = simulation.data_storage_dir
    pickled = tmp_path / "storage.pickle"
    pickled.write_bytes(
        pickle.dumps(simulation.persons._holders["disk_amount"]._disk_storage)
    )

    printed = _run(
        f"""
        import gc, pickle

        storage = pickle.loads(open({str(pickled)!r}, "rb").read())
        print(storage.get("2015").tolist())
        del storage
        gc.collect()
        """,
        tmp_path,
    )

    assert printed.split("\n")[0] == str(stored["disk_amount"].tolist())
    assert os.path.isdir(folder)
    _assert_reads(simulation, stored)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_process_collecting_the_simulation_leaves_the_folder(tmp_path):
    # Forked from a new interpreter that keeps to one thread (numexpr, which
    # pandas imports, otherwise starts a pool), so the fork is safe.
    printed = _run(
        """
        import gc, os
        import numpy as np
        from tests.fixtures.data_storage_dir import disk_simulation, read, values

        simulation = disk_simulation()
        simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
        folder = simulation.data_storage_dir
        pid = os.fork()
        if pid == 0:  # The child collects its copy of the simulation.
            status = 1
            try:
                del simulation
                gc.collect()
                status = 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        print(os.waitstatus_to_exitcode(status), os.path.isdir(folder))
        print(read(simulation, "disk_amount", "2015").tolist())
        """,
        tmp_path,
        NUMEXPR_MAX_THREADS="1",
    )

    assert printed.split("\n")[:2] == [
        "0 True",
        str(values("disk_amount", 1).tolist()),
    ]


# ----- A folder the caller chose ----- #


def test_a_folder_the_caller_chose_is_never_removed(tmp_path):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    simulation = disk_simulation(chosen)
    stored = _fill(simulation)
    clone = simulation.clone()
    clone.set_input("disk_amount", "2016", values("disk_amount", 5))

    assert simulation.data_storage_dir == str(chosen)
    # The clone stores in a folder of its own, made in the chosen one.
    assert Path(clone.data_storage_dir).parent.resolve() == chosen.resolve()
    _assert_reads(clone, stored)

    del simulation, clone
    gc.collect()

    # Everything the simulations made in it is removed; the folder stays.
    assert chosen.is_dir()
    assert list(chosen.iterdir()) == []


def test_a_disk_storage_never_removes_the_folder_containing_its_own(tmp_path):
    parent = tmp_path / "parent"
    storage_dir = parent / "storage"
    storage_dir.mkdir(parents=True)
    storage = OnDiskStorage(str(storage_dir))
    storage.put(np.arange(float(PEOPLE)), "2015")

    del storage
    gc.collect()

    assert not storage_dir.exists()
    assert parent.is_dir()


# ----- The folder object ----- #


def test_copies_of_the_folder_object_are_the_object():
    directory = TemporaryStorageDirectory()

    assert copy.copy(directory) is directory
    assert copy.deepcopy(directory) is directory
    assert pickle.loads(pickle.dumps(directory)) is directory
    assert directory.removes_directory


def test_a_folder_object_unpickled_after_removal_never_removes_the_path(tmp_path):
    directory = TemporaryStorageDirectory(str(tmp_path))
    pickled = pickle.dumps(directory)
    path = directory.path
    del directory
    gc.collect()
    assert not os.path.exists(path)
    # Something else now has a folder at that path.
    os.mkdir(path)

    unpickled = pickle.loads(pickled)
    del unpickled
    gc.collect()

    assert not pickle.loads(pickled).removes_directory
    assert os.path.isdir(path)
    os.rmdir(path)
