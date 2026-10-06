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
  process or in the one that pickled it. And a copied storage and its source
  wrote the same file for a key both stored after copying.
* A process forked from the simulation's did the same when it collected its
  copy of the simulation; and it made disk storages for new variables in the
  same subfolders as the process it was forked from, so each wrote over the
  other's files and removed them.
* The last disk storage collected removed the folder containing it once
  empty, even a folder the caller chose (``_data_storage_dir``).

Now a ``TemporaryStorageDirectory`` owns the folder a simulation makes, and
the simulation and every disk storage anywhere in the folder keep it alive;
it is removed when the last is collected or at interpreter exit, by the
process that made it, leaving only the subfolders of disk storages that
preserve theirs (``preserve_storage_dir``). A clone, or a forked process,
makes its own folder for the disk storages it makes. The properties over
any sequence of these operations are in ``test_data_storage_dir_property.py``.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import shutil
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
    STORE_ON_DISK,
    VARIABLES,
    Level,
    clone,
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
    cloned = clone(source)

    del source
    gc.collect()

    assert os.path.isdir(folder)
    _assert_reads(cloned, stored)
    del cloned
    gc.collect()
    assert not os.path.exists(folder)


def test_a_clone_of_a_branch_outliving_both_reads_every_value_on_disk():
    source = disk_simulation()
    stored = _fill(source)
    branch = source.get_branch("measurement")
    branch_stored = {"disk_amount": values("disk_amount", 7)}
    branch.set_input("disk_amount", "2016", branch_stored["disk_amount"])
    cloned = clone(branch)
    folder = source.data_storage_dir
    # The branch stored its value through the disk storage it copied, in the
    # source's folder: it made none of its own.
    assert branch._data_storage_dir is None

    del source, branch
    gc.collect()

    _assert_reads(cloned, stored)
    _assert_reads(cloned, branch_stored, period="2016")
    del cloned
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


@pytest.mark.parametrize("depth", [1, 2], ids=["subfolder", "nested"])
def test_a_disk_storage_made_in_the_folder_reads_its_values_after_the_simulation_is_collected(
    depth,
):
    simulation = disk_simulation()
    _fill(simulation)
    folder = simulation.data_storage_dir
    storage_dir = os.path.join(folder, *["mine"] * depth)
    os.makedirs(storage_dir)
    storage = OnDiskStorage(storage_dir)
    stored = values("disk_amount", 3)
    storage.put(stored, "2015")

    del simulation
    gc.collect()

    np.testing.assert_array_equal(storage.get("2015"), stored)
    del storage
    gc.collect()
    assert not os.path.exists(folder)


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


_FORK = """
    import gc, json, os, sys, traceback
    from tests.fixtures.data_storage_dir import disk_simulation, read, values

    def reads(simulation):
        return {{
            "folder": simulation.data_storage_dir,
            "disk_level 2015": read(simulation, "disk_level", "2015").tolist(),
            "disk_amount 2015": read(simulation, "disk_amount", "2015").tolist(),
            "disk_amount 2016": read(simulation, "disk_amount", "2016").tolist(),
        }}

    simulation = disk_simulation({chosen!r})
    folder = simulation.data_storage_dir
    simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
    parent_r, parent_w = os.pipe()
    child_r, child_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            clone = simulation.clone()
            # A variable with no holder yet, and a new period of one with one.
            simulation.set_input("disk_level", "2015", values("disk_level", 2))
            simulation.set_input("disk_amount", "2016", values("disk_amount", 3))
            clone.set_input("disk_level", "2016", values("disk_level", 7))
            os.write(child_w, b"x")
            os.read(parent_r, 1)
            result = reads(simulation)
            result["clone folder"] = clone.data_storage_dir
            result["clone disk_level 2016"] = read(clone, "disk_level", "2016").tolist()
            del simulation, clone
            gc.collect()
            os.write(child_w, json.dumps(result).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
        finally:
            os._exit(status)
    os.close(child_w)
    os.read(child_r, 1)
    # The same, once the child has, and a value stored before the fork.
    simulation.set_input("disk_level", "2015", values("disk_level", 4))
    simulation.set_input("disk_amount", "2016", values("disk_amount", 5))
    simulation.set_input("disk_amount", "2015", values("disk_amount", 6))
    os.write(parent_w, b"x")
    sent = b""
    while chunk := os.read(child_r, 65536):
        sent += chunk
    _, status = os.waitpid(pid, 0)
    print(json.dumps({{
        "status": os.waitstatus_to_exitcode(status),
        "child": json.loads(sent or "null"),
        "parent": reads(simulation),
        "parent folder": folder,
    }}))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize(
    "caller_chooses_the_folder", [False, True], ids=["made", "chosen"]
)
def test_a_forked_process_and_its_parent_keep_their_own_values(
    tmp_path, caller_chooses_the_folder
):
    chosen = None
    if caller_chooses_the_folder:
        chosen = tmp_path / "chosen"
        chosen.mkdir()
    printed = _run(
        _FORK.format(chosen=None if chosen is None else str(chosen)),
        tmp_path,
        NUMEXPR_MAX_THREADS="1",
    )
    result = json.loads(printed.splitlines()[-1])
    child, parent = result["child"], result["parent"]

    assert result["status"] == 0
    assert child["disk_level 2015"] == values("disk_level", 2).tolist()
    assert child["disk_amount 2016"] == values("disk_amount", 3).tolist()
    assert child["clone disk_level 2016"] == values("disk_level", 7).tolist()
    # What the parent stored again after the fork stays out of the child.
    assert child["disk_amount 2015"] == values("disk_amount", 1).tolist()
    assert parent["disk_level 2015"] == values("disk_level", 4).tolist()
    assert parent["disk_amount 2016"] == values("disk_amount", 5).tolist()
    assert parent["disk_amount 2015"] == values("disk_amount", 6).tolist()
    # The child, and a clone it made, stored their new variables in folders
    # of their own, inside the one the child inherited, and removed them when
    # they were collected.
    inherited = Path(result["parent folder"])
    assert parent["folder"] == str(inherited)
    for made in (child["folder"], child["clone folder"]):
        assert Path(made).parent.resolve() == inherited.resolve()
        assert not os.path.exists(made)


_FORK_THEN_PARENT_COLLECTS = """
    import gc, os, traceback
    from tests.fixtures.data_storage_dir import disk_simulation, read, values

    simulation = disk_simulation()
    simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
    folder = simulation.data_storage_dir
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    report_r, report_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            if {own!r}:
                # A variable with no holder yet: the child stores it in a
                # folder of its own, made inside the one it inherited.
                simulation.set_input("disk_level", "2015", values("disk_level", 2))
                assert os.path.dirname(simulation.data_storage_dir) == folder
            os.write(ready_w, b"x")
            os.read(go_r, 1)
            try:
                read(simulation, {variable!r}, "2015")
                reported = "read"
            except FileNotFoundError:
                reported = "FileNotFoundError"
            os.write(report_w, reported.encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(ready_w)
    os.close(report_w)
    os.read(ready_r, 1)
    del simulation
    gc.collect()
    removed = not os.path.exists(folder)
    os.write(go_w, b"x")
    reported = os.read(report_r, 64).decode() or "nothing"
    _, status = os.waitpid(pid, 0)
    print(os.waitstatus_to_exitcode(status), removed, reported)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.xfail(
    strict=True,
    # Only the documented failure counts: any other is a real one.
    raises=FileNotFoundError,
    reason=(
        "Documented limit: nothing in a forked process keeps alive the "
        "folder of the process it was forked from, and removing that "
        "folder removes the folder the forked process made inside it (see "
        "policyengine_core.data_storage.storage_directory)."
    ),
)
@pytest.mark.parametrize("own", [False, True], ids=["inherited", "its_own"])
def test_a_forked_process_reads_values_the_parent_has_collected(tmp_path, own):
    """A value the child inherited, or one it stored itself after forking."""
    printed = _run(
        _FORK_THEN_PARENT_COLLECTS.format(
            own=own, variable="disk_level" if own else "disk_amount"
        ),
        tmp_path,
        NUMEXPR_MAX_THREADS="1",
    )
    status, removed, reported = printed.split()[-3:]

    # The child ran to the end, and the parent removed its folder.
    assert (status, removed) == ("0", "True")
    if reported == "FileNotFoundError":
        raise FileNotFoundError(f"the child read no value ({own=})")
    assert reported == "read"


_RESTORED_FORK = """
    import os, traceback
    import numpy as np
    from policyengine_core.data_storage import OnDiskStorage

    folder = {folder!r}
    np.save(os.path.join(folder, "default_2015.npy"), np.array([1.0, 2.0]))
    storage = OnDiskStorage(folder, preserve_storage_dir=True)
    # A clone made before ``restore`` shares what the source reads back.
    early = storage.clone() if {writer!r} == "cloned_first" else None
    storage.restore()
    go_r, go_w = os.pipe()
    report_r, report_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.read(go_r, 1)
            os.write(report_w, repr(storage.get("2015").tolist()).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(report_w)
    if {deleted!r}:
        # Dropped since the fork: the child still reads it.
        storage.delete("2015")
    # The source writes, or a clone made after the fork, or one made before
    # ``restore``.
    writer = {{"source": storage, "cloned_first": early}}.get({writer!r}) or storage.clone()
    writer.put(np.array([8.0, 9.0]), "2015")
    os.write(go_w, b"x")
    child = os.read(report_r, 256).decode()
    _, status = os.waitpid(pid, 0)
    print(os.waitstatus_to_exitcode(status))
    print(child)
    print(repr(writer.get("2015").tolist()))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.parametrize("writer", ["source", "cloned", "cloned_first"])
@pytest.mark.parametrize("deleted", [False, True], ids=["kept", "deleted"])
def test_a_restored_storage_writes_over_no_file_a_forked_process_reads(
    tmp_path, deleted, writer
):
    """After forking, a storage, a clone made from it after the fork, or a
    clone made before it read the files back (``restore``) writes a new file
    for such a key rather than over the file the child reads, even after the
    source dropped that key."""
    folder = tmp_path / "restored"
    folder.mkdir()
    printed = _run(
        _RESTORED_FORK.format(folder=str(folder), deleted=deleted, writer=writer),
        tmp_path,
        NUMEXPR_MAX_THREADS="1",
    )

    assert printed.splitlines()[-3:] == ["0", "[1.0, 2.0]", "[8.0, 9.0]"]


# ----- Disk storages that preserve their folder ----- #


def _restored(storage_dir):
    """A new storage reading back the values stored in ``storage_dir``."""
    restored = OnDiskStorage(storage_dir, preserve_storage_dir=True)
    restored.restore()
    return restored


def _made_to_preserve_by_its_holder(simulation):
    # A holder that stores nothing on disk itself.
    simulation.memory_config = None
    holder = simulation.persons.get_holder("disk_amount")
    simulation.memory_config = STORE_ON_DISK
    return holder.create_disk_storage(preserve=True)


def _set_to_preserve(simulation):
    storage = simulation.persons.get_holder("disk_amount")._disk_storage
    storage.preserve_storage_dir = True
    return storage


def _made_to_preserve_in_the_folder(simulation):
    storage_dir = os.path.join(simulation.data_storage_dir, "mine", "disk_amount")
    os.makedirs(storage_dir)
    return OnDiskStorage(storage_dir, preserve_storage_dir=True)


@pytest.mark.parametrize(
    "preserving",
    [
        _made_to_preserve_by_its_holder,
        _set_to_preserve,
        _made_to_preserve_in_the_folder,
    ],
    ids=["create_disk_storage", "set", "made"],
)
def test_a_disk_storage_preserving_its_folder_keeps_it(preserving):
    simulation = disk_simulation()
    # Stored in a subfolder that goes with the folder.
    simulation.set_input("disk_level", "2015", values("disk_level", 0))
    folder = simulation.data_storage_dir
    storage = preserving(simulation)
    stored = values("disk_amount", 1)
    storage.put(stored, "2015")
    storage_dir = storage.storage_dir
    relative = Path(storage_dir).relative_to(folder).as_posix()
    try:
        del simulation, storage
        gc.collect()

        # Only the preserved folder is left, whole, with the folders leading
        # to it, and reads back as it was.
        assert _files(folder) == [f"{relative}/default_2015.npy"]
        np.testing.assert_array_equal(_restored(storage_dir).get("2015"), stored)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_a_disk_storage_preserving_its_folder_keeps_it_at_exit(tmp_path):
    printed = _run(
        """
        from tests.fixtures.data_storage_dir import SYSTEM, disk_simulation, values

        simulation = disk_simulation()
        SYSTEM.simulation = simulation
        simulation.set_input("disk_level", "2015", values("disk_level", 0))
        simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
        simulation.persons._holders["disk_amount"]._disk_storage.preserve_storage_dir = True
        print(simulation.data_storage_dir)
        """,
        tmp_path,
    )
    folder = printed.split()[-1]
    try:
        assert _files(folder) == ["disk_amount/default_2015.npy"]
        np.testing.assert_array_equal(
            _restored(os.path.join(folder, "disk_amount")).get("2015"),
            values("disk_amount", 1),
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_a_disk_storage_made_to_read_a_folder_another_removes_keeps_it_and_preserves_nothing():
    """It keeps the storage that removes the folder alive, and the folder
    goes once both are collected."""
    simulation = disk_simulation()
    simulation.set_input("disk_amount", "2015", values("disk_amount", 1))
    folder = simulation.data_storage_dir
    owner_dir = simulation.persons._holders["disk_amount"]._disk_storage.storage_dir
    reader = OnDiskStorage(owner_dir, preserve_storage_dir=True)
    reader.restore()

    del simulation
    gc.collect()

    np.testing.assert_array_equal(reader.get("2015"), values("disk_amount", 1))
    del reader
    gc.collect()
    assert not os.path.exists(folder)


# ----- A folder in one another simulation made ----- #


def _first_folder(first_stored):
    """A simulation and the folder it made, in which it stored a value or
    nothing."""
    first = disk_simulation()
    if first_stored:
        first.set_input("disk_level", "2015", values("disk_level", 0))
    return first, first.data_storage_dir


@pytest.mark.parametrize("first_stored", [True, False], ids=["stored", "made"])
def test_a_clone_of_a_simulation_given_another_simulations_folder_reads_its_values_after_that_one_is_collected(
    first_stored,
):
    first, folder = _first_folder(first_stored)
    given = disk_simulation(folder)
    # Cloned before it made any holder, so the clone makes a folder of its
    # own in the given one, owned by the clone.
    nested = clone(given)
    stored = {"disk_amount": values("disk_amount", 5)}
    nested.set_input("disk_amount", "2015", stored["disk_amount"])
    assert Path(nested.data_storage_dir).parent.resolve() == Path(folder).resolve()
    # A copy of its disk storage, which outlives the clone too.
    copied = copy.deepcopy(nested.persons._holders["disk_amount"]._disk_storage)
    del given

    del first
    gc.collect()

    _assert_reads(nested, stored)
    del nested
    gc.collect()
    np.testing.assert_array_equal(copied.get("2015"), stored["disk_amount"])
    del copied
    gc.collect()
    assert not os.path.exists(folder)


@pytest.mark.parametrize(
    "order", [("first", "clone"), ("clone", "first")], ids=["first", "clone"]
)
def test_a_disk_storage_preserving_its_folder_in_a_folder_made_in_another_simulations_keeps_it(
    order,
):
    first, folder = _first_folder(True)
    nested = clone(disk_simulation(folder))
    stored = values("disk_amount", 6)
    nested.set_input("disk_amount", "2015", stored)
    storage = nested.persons._holders["disk_amount"]._disk_storage
    storage.preserve_storage_dir = True
    storage_dir = storage.storage_dir
    relative = Path(storage_dir).relative_to(folder).as_posix()
    del storage
    simulations = {"first": first, "clone": nested}
    del first, nested
    try:
        for name in order:
            del simulations[name]
            gc.collect()

        # Only the preserved folder is left, whole, with the folders leading
        # to it, and reads back as it was.
        assert _files(folder) == [f"{relative}/default_2015.npy"]
        np.testing.assert_array_equal(_restored(storage_dir).get("2015"), stored)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize("first_stored", [True, False], ids=["stored", "made"])
@pytest.mark.parametrize("storing", ["given", "clone"])
def test_a_simulation_given_another_simulations_folder_stores_in_it_after_that_one_is_collected(
    first_stored, storing
):
    first, folder = _first_folder(first_stored)
    given = disk_simulation(folder)
    storer = given if storing == "given" else clone(given)
    del given

    del first
    gc.collect()

    stored = {"disk_amount": values("disk_amount", 7)}
    storer.set_input("disk_amount", "2015", stored["disk_amount"])
    _assert_reads(storer, stored)
    del storer
    gc.collect()
    assert not os.path.exists(folder)


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
