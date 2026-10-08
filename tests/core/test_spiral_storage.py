"""Exact spiral cleanup preserves inputs and independent storage snapshots.

Run the same deletion scenarios against memory and disk storage and compare
their observable values and provenance with explicit expected snapshots.
The shared-memory cases delete before reading the clone, exercising the
metadata cleanup without first materializing its shared arrays.
"""

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage


@pytest.fixture(params=["memory", "shared-memory", "disk"])
def storage_family(request, tmp_path):
    def make(is_eternal=False):
        if request.param == "disk":
            return OnDiskStorage(
                str(tmp_path),
                is_eternal=is_eternal,
                preserve_storage_dir=True,
            )
        return InMemoryStorage(is_eternal=is_eternal)

    def clone(storage):
        if isinstance(storage, InMemoryStorage):
            return storage.clone(share_arrays=request.param == "shared-memory")
        return storage.clone()

    return make, clone


def _put(storage, period, branch, value, derived=False):
    storage.put(np.array([value]), period, branch, derived=derived)


def _assert_snapshot(storage, expected, probes):
    """Check values, input provenance and keys against the expected snapshot.

    ``expected`` maps (branch, period) to (value, derived), using storage's
    public period representation rather than its encoded internal keys.
    """
    expected = {
        (branch, periods.period(period)): (value, derived)
        for (branch, period), (value, derived) in expected.items()
    }
    assert set(storage.get_known_branch_periods()) == set(expected)
    for branch, period in probes:
        canonical = periods.period(periods.ETERNITY if storage.is_eternal else period)
        entry = expected.get((branch, canonical))
        assert storage.has(period, branch) == (entry is not None)
        assert storage.is_derived(period, branch) == (entry is not None and entry[1])
        if entry is None:
            assert storage.get(period, branch) is None
        else:
            np.testing.assert_array_equal(storage.get(period, branch), [entry[0]])


def _assert_metadata(storage):
    """Deleted entries retain neither sharing nor provenance marks."""
    if isinstance(storage, InMemoryStorage):
        assert storage._shared <= storage._arrays.keys()
        assert storage._inputs <= storage._arrays.keys()
    else:
        assert storage._derived <= storage._files.keys()


def test_exact_cleanup_keeps_descendant_inputs_and_contained_periods(storage_family):
    make, clone = storage_family
    source = make()
    original = {
        ("default", "2021"): (10, True),
        ("default", "2021-01"): (20, False),
        ("other", "2021"): (30, True),
    }
    for (branch, period), (value, derived) in original.items():
        _put(source, period, branch, value, derived)
    copy = clone(source)
    descendant = clone(copy)
    # A descendant replaces an inherited calculation under the same name
    # with its own input. Cleanup must protect that input independently.
    _put(descendant, "2021", "default", 100)

    source.delete_exact("2021", "default", derived_only=True)
    _assert_metadata(source)
    remaining = dict(original)
    del remaining[("default", "2021")]
    probes = list(original)
    _assert_snapshot(source, remaining, probes)
    _assert_snapshot(copy, original, probes)

    copy.delete_exact("2021", "default", derived_only=True)
    descendant.delete_exact("2021", "default", derived_only=True)
    _assert_metadata(copy)
    _assert_metadata(descendant)
    _assert_snapshot(copy, remaining, probes)
    own_input = dict(original)
    own_input[("default", "2021")] = (100, False)
    _assert_snapshot(descendant, own_input, probes)


def test_exact_deletion_releases_clone_marks_without_changing_its_source(
    storage_family,
):
    make, clone = storage_family
    source = make()
    _put(source, "2021", "default", 10)
    _put(source, "2022", "default", 20, derived=True)
    copy = clone(source)
    descendant = clone(copy)
    probes = [("default", "2021"), ("default", "2022")]
    original = {probes[0]: (10, False), probes[1]: (20, True)}

    # Unconditional deletion still removes inputs, and dropping every
    # shared key must work with the empty-set sentinels on a later retry.
    copy.delete_exact("2021")
    copy.delete_exact("2022", derived_only=True)
    copy.delete_exact("2021")
    copy.delete_exact("2022", derived_only=True)
    _assert_metadata(copy)
    if isinstance(copy, InMemoryStorage):
        assert "_inputs" not in vars(copy)
        assert "_shared" not in vars(copy)
    _assert_snapshot(copy, {}, probes)
    _assert_snapshot(source, original, probes)
    _assert_snapshot(descendant, original, probes)

    # Reusing deleted keys changes only this copy, including provenance;
    # disk storage must not overwrite files the other snapshots still read.
    _put(copy, "2021", "default", 101, derived=True)
    _put(copy, "2022", "default", 202)
    _assert_snapshot(copy, {probes[0]: (101, True), probes[1]: (202, False)}, probes)
    _assert_snapshot(source, original, probes)
    _assert_snapshot(descendant, original, probes)


def test_derived_only_cleanup_keeps_an_input_and_its_clone_metadata(storage_family):
    make, clone = storage_family
    source = make()
    _put(source, "2021", "default", 10)
    copy = clone(source)
    copy.delete_exact("2021", derived_only=True)
    _assert_metadata(copy)

    expected = {("default", "2021"): (10, False)}
    probes = list(expected)
    _assert_snapshot(copy, expected, probes)
    _assert_snapshot(source, expected, probes)
    # A protected input can still be removed explicitly from the copy.
    copy.delete_exact("2021")
    _assert_metadata(copy)
    _assert_snapshot(copy, {}, probes)
    _assert_snapshot(source, expected, probes)


def test_eternal_exact_cleanup_normalizes_the_period_and_isolates_branches(
    storage_family,
):
    make, clone = storage_family
    source = make(is_eternal=True)
    _put(source, "2021", "default", 10)
    _put(source, "2021-01", "other", 20, derived=True)
    copy = clone(source)
    original = {
        ("default", periods.ETERNITY): (10, False),
        ("other", periods.ETERNITY): (20, True),
    }
    probes = [
        (branch, period)
        for branch in ("default", "other")
        for period in ("2021", "2022-02", periods.ETERNITY)
    ]

    copy.delete_exact("2022-02", "default", derived_only=True)
    copy.delete_exact("2022", "other", derived_only=True)
    _assert_metadata(copy)
    _assert_snapshot(copy, {("default", periods.ETERNITY): (10, False)}, probes)
    _assert_snapshot(source, original, probes)
    copy.delete_exact("2023-03", "default")
    _assert_metadata(copy)
    _assert_snapshot(copy, {}, probes)
    _assert_snapshot(source, original, probes)
