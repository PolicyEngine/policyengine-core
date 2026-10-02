"""Disk and memory storage keep the same values through any puts and deletes.

Random sequences of ``put``, ``delete(period, branch_name)`` and
``delete(None, branch_name)`` run on an ``InMemoryStorage`` and an
``OnDiskStorage``, and on a reference model of the documented behaviour:
``delete(period, branch_name)`` removes the branch's values for every period
that ``period`` contains, ``delete(None, branch_name)`` removes all of the
branch's values, and eternal storage keeps one ``ETERNITY`` value per branch.
After every step both storages hold exactly the model's (branch, period)
keys, with the values last put there.

Branch names include ones that contain ``_`` or start with another branch's
name followed by ``_``, since disk keys join branch and period with ``_``.
``test_disk_storage_delete_contained_periods.py`` pins the same behaviour
with examples.
"""

import os
import tempfile

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage

BRANCH_NAMES = (
    "default",
    "reform",
    "pre",
    "pre_tcja",
    "pre_tcja_ctc",
    "y",
    "y_2019",
    "trailing_",
)


@st.composite
def period_strategy(draw):
    if draw(st.integers(0, 19)) == 0:
        return periods.period(periods.ETERNITY)
    unit = draw(st.sampled_from([periods.YEAR, periods.MONTH, periods.DAY]))
    year = draw(st.integers(2023, 2026))
    if os.name == "nt":
        # Windows file names cannot contain the ``:`` of a multi-unit or
        # rolling-year period's string form.
        month = 1 if unit == periods.YEAR else draw(st.integers(1, 12))
        size = 1
    else:
        month = draw(st.integers(1, 12))
        size = draw(st.integers(1, 3))
    # In-memory storage rejects years and months anchored mid-month, whose
    # string form drops the day.
    day = draw(st.integers(1, 28)) if unit == periods.DAY else 1
    return periods.Period((unit, periods.Instant((year, month, day)), size))


put_step = st.tuples(
    st.just("put"),
    period_strategy(),
    st.sampled_from(BRANCH_NAMES),
    st.integers(0, 1_000),
)
delete_step = st.tuples(
    st.just("delete"),
    st.one_of(st.none(), period_strategy()),
    st.sampled_from(BRANCH_NAMES),
    st.none(),
)


def stored(storage):
    """Map each stored (branch name, period string) to its value."""
    return {
        (branch_name, str(period)): float(storage.get(period, branch_name)[0])
        for branch_name, period in storage.get_known_branch_periods()
    }


@hypothesis.settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(
    is_eternal=st.booleans(),
    steps=st.lists(st.one_of(put_step, put_step, delete_step), max_size=25),
)
def test_disk_and_memory_storage_delete_the_same_keys(is_eternal, steps):
    model = {}
    memory = InMemoryStorage(is_eternal=is_eternal)
    with tempfile.TemporaryDirectory() as directory:
        disk = OnDiskStorage(
            directory, is_eternal=is_eternal, preserve_storage_dir=True
        )
        for action, period, branch_name, value in steps:
            if period is not None and is_eternal:
                period = periods.period(periods.ETERNITY)
            if action == "put":
                memory.put(np.array([float(value)]), period, branch_name)
                disk.put(np.array([float(value)]), period, branch_name)
                model[(branch_name, str(period))] = float(value)
            else:
                memory.delete(period, branch_name)
                disk.delete(period, branch_name)
                model = {
                    (key_branch_name, key_period): key_value
                    for (key_branch_name, key_period), key_value in model.items()
                    if key_branch_name != branch_name
                    or (
                        period is not None
                        and not period.contains(periods.period(key_period))
                    )
                }

            assert stored(disk) == stored(memory) == model
