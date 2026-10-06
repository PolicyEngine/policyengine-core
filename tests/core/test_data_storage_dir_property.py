"""Properties of the folder a simulation stores values on disk in.

Over any sequence of these operations on simulations that store every value
on disk, starting from one simulation:

* ``set``: a simulation stores a value (``set_input``), or a disk storage on
  its own does (``put``);
* ``set_every``: every simulation stores a different value of every
  variable for one year, and every disk storage on its own one for that
  year, so any two that share files both store a key neither may have had;
* ``clone`` and ``branch``: ``Simulation.clone`` (through the fixture's
  ``clone``, so the clone does not keep its source alive) and
  ``get_branch``;
* ``copy``: a disk storage of a simulation, or one on its own, is pickled or
  copied (``copy.deepcopy`` or ``copy.copy``), giving a disk storage on its
  own;
* ``make``: a disk storage on its own is made in a new subfolder of a
  simulation's folder, preserving it or not (``preserve_storage_dir``);
* ``nest``: a new simulation is given a simulation's folder: a new
  subfolder of it, or the folder itself, in which case only its clone,
  made before it stores anything, is kept (two simulations storing a
  variable in one folder would share its subfolder). Its clones then make
  their folders in a folder another simulation may have made;
* ``preserve``: a disk storage a simulation or ``make`` made is set to
  preserve its folder;
* ``collect``: one simulation or disk storage on its own is dropped and the
  garbage collector run, in any order, so sources go before their clones
  and copies, or after (a branch keeps its source alive);

with the first simulation's folder made by the simulation or chosen by the
caller (``_data_storage_dir``):

1. Nothing removes or writes over a file anything still reads: after every
   collection, each simulation left reads every value it stored or inherited
   when cloned or branched, and each disk storage on its own every value it
   was copied with or stored.
2. Only what is preserved is left: once everything is collected, the only
   folders left in the temporary folder or in the caller's are the subfolders
   of disk storages that preserve theirs and the folders leading to them, and
   each still holds the files its storage last read, unchanged.
3. A folder the caller chose is never removed, unless it is in a folder a
   simulation made: that one removes it with everything else in it.

Regressions are in ``test_data_storage_dir.py``.
"""

from __future__ import annotations

import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

import copy  # noqa: E402
import gc  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
import numpy as np  # noqa: E402

from policyengine_core.data_storage import OnDiskStorage  # noqa: E402
from tests.fixtures.data_storage_dir import (  # noqa: E402
    VARIABLES,
    YEARS,
    clone,
    disk_simulation,
    read,
    temporary_folders,
    values,
)

COPIERS = {
    "pickle": lambda storage: pickle.loads(pickle.dumps(storage)),
    "deepcopy": copy.deepcopy,
    "copy": copy.copy,
}

_index = st.integers(min_value=0, max_value=63)
OPERATIONS = st.lists(
    st.one_of(
        st.tuples(
            st.just("set"),
            _index,
            st.sampled_from(VARIABLES),
            st.sampled_from(YEARS),
            st.integers(min_value=0, max_value=1000),
        ),
        st.tuples(
            st.just("set_every"),
            st.sampled_from(YEARS),
            st.integers(min_value=0, max_value=1000),
        ),
        st.tuples(st.just("clone"), _index),
        st.tuples(st.just("branch"), _index),
        st.tuples(
            st.just("copy"),
            _index,
            st.sampled_from(VARIABLES),
            st.sampled_from(sorted(COPIERS)),
        ),
        st.tuples(st.just("make"), _index, st.booleans()),
        st.tuples(st.just("nest"), _index, st.booleans()),
        st.tuples(st.just("preserve"), _index, st.sampled_from(VARIABLES)),
        st.tuples(st.just("collect"), _index),
    ),
    max_size=24,
)


def _contents(storage) -> dict:
    """What ``storage`` reads: ``{storage key: values}``."""
    return {
        key: np.asarray(storage.get(*reversed(key.rsplit("_", 1))))
        for key in storage._files
    }


@dataclass
class _Simulation:
    """A simulation, and what it must read: ``{(variable, year): values}``."""

    simulation: object
    expected: dict = field(default_factory=dict)
    # The variables whose disk storage this simulation made (rather than
    # copied from its source), and those of them set to preserve their folder.
    made: set = field(default_factory=set)
    preserved: set = field(default_factory=set)

    def check(self):
        for (variable, year), expected in self.expected.items():
            np.testing.assert_array_equal(
                np.asarray(read(self.simulation, variable, year)), expected
            )

    variables = VARIABLES

    def set(self, variable, year, stored):
        if variable not in self.simulation.persons._holders:
            self.made.add(variable)
        self.simulation.set_input(variable, year, stored)
        self.expected[(variable, year)] = stored

    def disk_storage(self, variable):
        holder = self.simulation.persons._holders.get(variable)
        return None if holder is None else holder._disk_storage

    def preserved_storages(self):
        return [self.disk_storage(variable) for variable in self.preserved]


@dataclass
class _Storage:
    """A disk storage on its own, and what it must read: ``{storage key:
    values}``."""

    storage: object
    expected: dict
    # Whether ``make`` made it (rather than copying it), and whether it is set
    # to preserve its folder.
    made: bool = False
    preserved: bool = False

    def check(self):
        for key, expected in self.expected.items():
            branch_name, period = key.rsplit("_", 1)
            np.testing.assert_array_equal(
                np.asarray(self.storage.get(period, branch_name)), expected
            )

    # What it stores has no variable: values like this one's.
    variables = VARIABLES[:1]

    def set(self, variable, year, stored):
        self.storage.put(stored, year)
        self.expected[f"default_{year}"] = stored

    def disk_storage(self, variable):
        return self.storage

    def preserved_storages(self):
        return [self.storage] if self.preserved else []


class _Run:
    """Operations applied to what they made, which only ``live`` holds, so
    that dropping an item from it is all that keeps the item from being
    collected."""

    def __init__(self, chosen_folder):
        self.live = [_Simulation(disk_simulation(chosen_folder))]
        self.made = 0
        # The folders ``nest`` gave simulations, as a caller would.
        self.given = []
        # The files each preserved storage last read: ``{storage directory:
        # {path: values}}``.
        self.preserved = {}

    def apply(self, operation):
        if operation[0] == "set_every":
            _, year, seed = operation
            for offset, item in enumerate(self.live):
                for variable in item.variables:
                    item.set(variable, year, values(variable, seed + offset))
            self._record_preserved()
            return
        kind, index = operation[0], operation[1] % len(self.live)
        if kind == "collect":
            del self.live[index]
            gc.collect()
            for item in self.live:
                item.check()
            return
        target = self.live[index]
        if kind == "set":
            _, _, variable, year, seed = operation
            target.set(variable, year, values(variable, seed))
        elif kind == "copy":
            _, _, variable, how = operation
            storage = target.disk_storage(variable)
            if storage is not None:
                self.live.append(_Storage(COPIERS[how](storage), _contents(storage)))
        elif not isinstance(target, _Simulation):
            if kind == "preserve" and target.made:
                target.storage.preserve_storage_dir = True
                target.preserved = True
        elif kind == "clone":
            self.live.append(
                _Simulation(clone(target.simulation), dict(target.expected))
            )
        elif kind == "branch":
            self.made += 1
            branch = target.simulation.get_branch(f"branch_{self.made}")
            self.live.append(_Simulation(branch, dict(target.expected)))
        elif kind == "make":
            _, _, preserve = operation
            self.made += 1
            storage_dir = os.path.join(
                target.simulation.data_storage_dir, f"made_{self.made}"
            )
            os.mkdir(storage_dir)
            storage = OnDiskStorage(storage_dir, preserve_storage_dir=preserve)
            stored = values(VARIABLES[0], self.made)
            storage.put(stored, YEARS[0])
            self.live.append(
                _Storage(
                    storage,
                    {f"default_{YEARS[0]}": stored},
                    made=True,
                    preserved=preserve,
                )
            )
        elif kind == "nest":
            _, _, inside = operation
            folder = target.simulation.data_storage_dir
            if inside:
                self.made += 1
                folder = os.path.join(folder, f"nested_{self.made}")
                os.mkdir(folder)
                self.given.append(folder)
                nested = disk_simulation(folder)
            else:
                nested = clone(disk_simulation(folder))
            self.live.append(_Simulation(nested))
        elif kind == "preserve":
            _, _, variable = operation
            if variable in target.made:
                target.disk_storage(variable).preserve_storage_dir = True
                target.preserved.add(variable)
        self._record_preserved()

    def _record_preserved(self):
        for item in self.live:
            for storage in item.preserved_storages():
                self.preserved[storage.storage_dir] = {
                    path: np.load(path) for path in storage._files.values()
                }


def _run(operations, chosen_folder) -> tuple:
    """Apply ``operations``, check what is left reads its values, and return
    the files preserved storages last read, and the folders ``nest`` gave
    simulations."""
    run = _Run(chosen_folder)
    for operation in operations:
        if not run.live:
            break
        run.apply(operation)
    for item in run.live:
        item.check()
    return run.preserved, run.given


def _check_left(root, given, preserved):
    """Only the folders given to simulations that are in no folder a
    simulation made, the preserved folders, and the folders leading to them
    are left in ``root``; the preserved folders hold the files their storages
    last read."""
    kept = {os.path.realpath(storage_dir) for storage_dir in preserved}
    leading = {os.path.realpath(root)}
    for folder in given:
        if not any(
            part.startswith("openfisca_")
            for part in Path(os.path.relpath(folder, root)).parts
        ):
            assert os.path.isdir(folder), f"{folder}, given, is removed"
            leading.add(os.path.realpath(folder))
    for folder, _, files in os.walk(root):
        real = os.path.realpath(folder)
        if any(real == path or real.startswith(path + os.sep) for path in kept):
            continue
        assert real in leading or any(
            path.startswith(real + os.sep) for path in kept
        ), f"{folder} is left"
        assert files == [], f"{files} are left in {folder}"
    for files in preserved.values():
        for path, expected in files.items():
            np.testing.assert_array_equal(np.load(path), expected)


@settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations=OPERATIONS, caller_chooses_the_folder=st.booleans())
# Counterexamples from before a simulation kept alive the folder its own is
# in: a simulation given a subfolder of another's folder, or the clone of
# one given the folder itself, stores a new variable once the other is
# collected. The derandomized examples need not include either.
@example(
    operations=[
        ("nest", 0, True),
        ("collect", 0),
        ("set", 0, VARIABLES[0], YEARS[0], 0),
    ],
    caller_chooses_the_folder=False,
)
@example(
    operations=[
        ("nest", 0, False),
        ("collect", 0),
        ("set", 0, VARIABLES[0], YEARS[0], 0),
    ],
    caller_chooses_the_folder=False,
)
def test_storage_folders_keep_what_is_read_and_leave_only_what_is_preserved(
    operations, caller_chooses_the_folder
):
    with temporary_folders() as root:
        chosen = None
        if caller_chooses_the_folder:
            chosen = os.path.join(root, "chosen")
            os.mkdir(chosen)

        preserved, given = _run(operations, chosen)
        gc.collect()

        if chosen is not None:
            assert os.path.isdir(chosen)
            given.append(chosen)
        _check_left(root, given, preserved)
