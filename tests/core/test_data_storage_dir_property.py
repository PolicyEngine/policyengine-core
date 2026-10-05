"""Properties of the folder a simulation stores values on disk in.

Over any sequence of these operations on simulations that store every value
on disk, starting from one simulation:

* ``set``: a simulation stores a value (``set_input``);
* ``clone`` and ``branch``: ``Simulation.clone`` and ``get_branch``;
* ``copy``: a disk storage of a simulation is pickled or deep-copied;
* ``collect``: one simulation or copied storage is dropped and the garbage
  collector run, in any order, so sources go before their clones, branches
  and copies, or after;

with the first simulation's folder made by the simulation or chosen by the
caller (``_data_storage_dir``):

1. Nothing removes a file anything still reads: after every collection, each
   simulation left reads every value it stored or inherited when cloned or
   branched, and each copied storage every value its source had.
2. No folder is left: once everything is collected, no ``openfisca_*`` folder
   is left in the temporary folder or in the caller's.
3. A folder the caller chose is never removed (everything made in it is).

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

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
import numpy as np  # noqa: E402

from tests.fixtures.data_storage_dir import (  # noqa: E402
    VARIABLES,
    YEARS,
    disk_simulation,
    read,
    storage_folders,
    temporary_folders,
    values,
)

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
        st.tuples(st.just("clone"), _index),
        st.tuples(st.just("branch"), _index),
        st.tuples(
            st.just("copy"),
            _index,
            st.sampled_from(VARIABLES),
            st.sampled_from(["pickle", "deepcopy"]),
        ),
        st.tuples(st.just("collect"), _index),
    ),
    max_size=24,
)


@dataclass
class _Simulation:
    """A simulation, and what it must read: ``{(variable, year): values}``."""

    simulation: object
    expected: dict = field(default_factory=dict)

    def check(self):
        for (variable, year), expected in self.expected.items():
            np.testing.assert_array_equal(
                np.asarray(read(self.simulation, variable, year)), expected
            )


@dataclass
class _Copy:
    """A copied disk storage, and what it must read: ``{storage key: values}``."""

    storage: object
    expected: dict

    def check(self):
        for key, expected in self.expected.items():
            branch_name, period = key.rsplit("_", 1)
            np.testing.assert_array_equal(
                np.asarray(self.storage.get(period, branch_name)), expected
            )


def _copy(simulation, variable, how):
    """A copy of ``simulation``'s disk storage of ``variable``, if it has one."""
    holder = simulation.persons._holders.get(variable)
    if holder is None or holder._disk_storage is None:
        return None
    storage = holder._disk_storage
    expected = {
        key: np.asarray(storage.get(*reversed(key.rsplit("_", 1))))
        for key in storage._files
    }
    copier = copy.deepcopy if how == "deepcopy" else _pickled
    return _Copy(copier(storage), expected)


def _pickled(storage):
    return pickle.loads(pickle.dumps(storage))


def _apply(live, operation, branches):
    """Apply one operation to ``live``. Returns the number of branches made."""
    kind, index = operation[0], operation[1] % len(live)
    if kind == "collect":
        del live[index]
        gc.collect()
        for item in live:
            item.check()
        return branches
    target = live[index]
    if not isinstance(target, _Simulation):
        return branches
    simulation = target.simulation
    if kind == "set":
        _, _, variable, year, seed = operation
        stored = values(variable, seed)
        simulation.set_input(variable, year, stored)
        target.expected[(variable, year)] = stored
    elif kind == "clone":
        live.append(_Simulation(simulation.clone(), dict(target.expected)))
    elif kind == "branch":
        branches += 1
        branch = simulation.get_branch(f"branch_{branches}")
        live.append(_Simulation(branch, dict(target.expected)))
    else:
        copied = _copy(simulation, *operation[2:])
        if copied is not None:
            live.append(copied)
    return branches


def _run(operations, chosen_folder):
    """Apply ``operations``, then check what is left reads its values."""
    # Only ``live`` holds what the operations make, so that dropping an item
    # from it is all that keeps the item from being collected.
    live = [_Simulation(disk_simulation(chosen_folder))]
    branches = 0
    for operation in operations:
        if not live:
            break
        branches = _apply(live, operation, branches)
    for item in live:
        item.check()


@settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(operations=OPERATIONS, caller_chooses_the_folder=st.booleans())
def test_storage_folders_last_exactly_as_long_as_their_readers(
    operations, caller_chooses_the_folder
):
    with temporary_folders() as root:
        chosen = None
        if caller_chooses_the_folder:
            chosen = os.path.join(root, "chosen")
            os.mkdir(chosen)

        _run(operations, chosen)
        gc.collect()

        assert storage_folders(root) == []
        if chosen is not None:
            assert os.path.isdir(chosen)
            assert os.listdir(chosen) == []
