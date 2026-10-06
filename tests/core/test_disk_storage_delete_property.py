"""Disk and memory storage keep the same values through any puts and deletes.

Random sequences of ``put``, ``delete(period, branch_name)`` and
``delete(None, branch_name)`` run on an ``InMemoryStorage`` and an
``OnDiskStorage``, and on a reference model of the documented behaviour:
``put`` stores a value unless no key can hold its branch name and period (a
branch name containing ``:``, or a period whose string form names a period
starting on another day: a year or month anchored mid-month), and then both
storages reject it, with the same error; ``delete(period, branch_name)``
removes the branch's values for every period that ``period`` contains,
``delete(None, branch_name)`` removes all of the branch's values, and eternal
storage keeps one ``ETERNITY`` value per branch. After every step both
storages hold exactly the model's (branch, period) keys, with the values last
put there.

A second property restores a disk storage's directory, holding the files of
random keys and of names that are not keys, and checks that ``restore`` reads
back exactly the keys, warns about the other names, and that deletes then
agree with memory.

Branch names include ones that contain ``_`` or start with another branch's
name followed by ``_``, since disk keys join branch and period with ``_``.
``test_disk_storage_delete_contained_periods.py`` pins the same behaviour
with examples.
"""

import os
import tempfile
import warnings

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
    # No key can hold a branch name containing ``:``.
    "my:reform",
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
    # Both storages reject years and months anchored mid-month, whose string
    # form drops the day (before writing a file, so on Windows too).
    if unit == periods.DAY:
        day = draw(st.integers(1, 28))
    else:
        day = draw(st.one_of(st.just(1), st.integers(2, 28)))
    return periods.Period((unit, periods.Instant((year, month, day)), size))


def storable(branch_name, period):
    """The model of which keys a storage holds: those whose branch name has no
    ``:`` and whose period's string form names a period starting the same
    day."""
    return (
        ":" not in branch_name
        and periods.period(str(period)).start == periods.period(period).start
    )


def put_both(memory, disk, value, period, branch_name):
    """Put ``value`` in both storages; whether both stored it. Both store it,
    or both raise the same ``ValueError``, as the model says."""
    errors = []
    for storage in (memory, disk):
        try:
            storage.put(np.array([float(value)]), period, branch_name)
            errors.append(None)
        except ValueError as error:
            errors.append(str(error))
    assert errors[0] == errors[1]
    if memory.is_eternal:
        period = periods.period(periods.ETERNITY)
    assert (errors[0] is None) == storable(branch_name, period)
    return errors[0] is None


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
                if put_both(memory, disk, value, period, branch_name):
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


# Names of ``.npy`` files that are not keys: no ``_``, no period after the last
# ``_``, or a period not in its string form. (Not ``eternity``: on a
# case-insensitive file system that is the ``ETERNITY`` key's file.)
NON_CANONICAL_PERIODS = ["2025-3", "2025-03-1", "x", ""]
if os.name != "nt":
    NON_CANONICAL_PERIODS += ["year:2025", "month:2025-01:12", "day:2025-03-01"]
not_key_names = st.one_of(
    st.text(alphabet="abcxyz0123456789-", min_size=1, max_size=8),
    st.builds(
        "{}_{}".format,
        st.sampled_from([name for name in BRANCH_NAMES if ":" not in name]),
        st.one_of(
            st.sampled_from(NON_CANONICAL_PERIODS),
            st.text(alphabet="abcxyz", min_size=1, max_size=6),
        ),
    ),
).map(lambda name: name + ".npy")


@hypothesis.settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(
    is_eternal=st.booleans(),
    puts=st.lists(put_step, max_size=12),
    not_keys=st.lists(not_key_names, max_size=4, unique=True),
    deletes=st.lists(delete_step, max_size=4),
)
def test_restore_reads_back_the_keys_and_only_the_keys(
    is_eternal, puts, not_keys, deletes
):
    memory = InMemoryStorage(is_eternal=is_eternal)
    with tempfile.TemporaryDirectory() as directory:
        disk = OnDiskStorage(
            directory, is_eternal=is_eternal, preserve_storage_dir=True
        )
        for _, period, branch_name, value in puts:
            put_both(memory, disk, value, period, branch_name)
        for name in not_keys:
            np.save(os.path.join(directory, name), np.array([-1.0]))

        restored = OnDiskStorage(
            directory, is_eternal=is_eternal, preserve_storage_dir=True
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.restore()

        assert stored(restored) == stored(memory)
        if not_keys:
            (warning,) = caught
            for name in not_keys:
                assert name in str(warning.message)
        else:
            assert caught == []

        for _, period, branch_name, _ in deletes:
            memory.delete(period, branch_name)
            restored.delete(period, branch_name)
            assert stored(restored) == stored(memory)
