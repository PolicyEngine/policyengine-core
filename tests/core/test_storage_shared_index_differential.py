"""A storage without a set of its own behaves like one that always had one.

``_EagerIndexStorage`` below is ``InMemoryStorage`` as it was when every
storage owned a set of shared keys. Random sequences of operations run on
both, and after every step the two must hold the same arrays and the same
shared keys; at the end every read must agree. The storage under test must
also keep its own invariants: it refers to the one shared-nothing object
exactly when it shares nothing, and no two storages hold the same set.
``test_storage_shared_index.py`` pins the same behaviour with examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.in_memory_storage import (
    _NOTHING_SHARED,
    _can_share,
    _read_only_view,
)


class _EagerIndexStorage:
    """The storage with an empty set of shared keys in every instance."""

    def __init__(self, is_eternal):
        self._arrays = {}
        self._shared = set()
        self.is_eternal = is_eternal

    def clone(self, share_arrays=False):
        clone = _EagerIndexStorage(self.is_eternal)
        if share_arrays:
            for key, array in self._arrays.items():
                if _can_share(array):
                    clone._arrays[key] = _read_only_view(array)
                    clone._shared.add(key)
                else:
                    clone._arrays[key] = array.copy()
        else:
            clone._arrays = {key: array.copy() for key, array in self._arrays.items()}
        return clone

    def get(self, period, branch_name="default"):
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        key = f"{branch_name}:{period}"
        values = self._arrays.get(key)
        if values is None:
            return None
        if key in self._shared:
            values = values.copy()
            self._arrays[key] = values
            self._shared.discard(key)
        return values

    def put(self, value, period, branch_name="default"):
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        key = f"{branch_name}:{period}"
        self._arrays[key] = value
        self._shared.discard(key)

    def delete(self, period=None, branch_name="default"):
        if period is None:
            branch_prefix = f"{branch_name}:"
            self._arrays = {
                period_item: value
                for period_item, value in self._arrays.items()
                if not period_item.startswith(branch_prefix)
            }
            self._shared.intersection_update(self._arrays)
            return
        if self.is_eternal:
            period = periods.period(periods.ETERNITY)
        period = periods.period(period)
        self._arrays = {
            period_item: value
            for period_item, value in self._arrays.items()
            if not (
                period_item.startswith(f"{branch_name}:")
                and period.contains(periods.period(period_item.split(":", 1)[1]))
            )
        }
        self._shared.intersection_update(self._arrays)


PERIODS = ["2017-01", "2017-02", "2017"]
BRANCHES = ["default", "other"]
ARRAY_KINDS = ["plain", "read_only", "masked"]


def _make_array(kind, seed):
    values = np.array([seed, seed + 1.0, seed + 2.0])
    if kind == "read_only":
        values.flags.writeable = False
    if kind == "masked":
        return np.ma.masked_array(values, mask=[False, True, False])
    return values


_storage = st.integers(min_value=0, max_value=10**6)
_period = st.sampled_from(PERIODS)
_branch = st.sampled_from(BRANCHES)
_seed = st.integers(min_value=0, max_value=50).map(float)

_operation = st.one_of(
    st.tuples(
        st.just("put"),
        _storage,
        _period,
        _branch,
        st.sampled_from(ARRAY_KINDS),
        _seed,
    ),
    st.tuples(st.just("get"), _storage, _period, _branch),
    st.tuples(st.just("write"), _storage, _period, _branch, _seed),
    st.tuples(st.just("delete"), _storage, st.none() | _period, _branch),
    st.tuples(st.just("clone"), _storage, st.booleans()),
    st.tuples(st.just("clear"), _storage),
)


def _same_array(tested, reference):
    if tested is None or reference is None:
        return tested is None and reference is None
    if type(tested) is not type(reference) or tested.dtype != reference.dtype:
        return False
    if isinstance(tested, np.ma.MaskedArray):
        return np.array_equal(tested.mask, reference.mask) and np.array_equal(
            tested.data, reference.data
        )
    return np.array_equal(tested, reference)


def _apply(storages, operation):
    """Run one operation on a list of storages and return what it read."""
    kind, index = operation[0], operation[1] % len(storages)
    storage = storages[index]
    if kind == "put":
        _, _, period, branch, array_kind, seed = operation
        storage.put(_make_array(array_kind, seed), period, branch)
        return None
    if kind == "get":
        _, _, period, branch = operation
        return storage.get(period, branch)
    if kind == "write":
        # Write in place into what a read returned, as formulas do.
        _, _, period, branch, seed = operation
        values = storage.get(period, branch)
        if values is not None and values.flags.writeable:
            values[0] = seed
        return values
    if kind == "delete":
        _, _, period, branch = operation
        storage.delete(period, branch)
        return None
    if kind == "clone":
        storages.append(storage.clone(share_arrays=operation[2]))
        return None
    if kind == "clear":
        # policyengine-uk clears a dropped branch's storage like this.
        storage._arrays.clear()
        return None
    raise AssertionError(kind)


def _assert_same_state(tested, reference):
    assert len(tested) == len(reference)
    for index, (storage, expected) in enumerate(zip(tested, reference)):
        assert storage._arrays.keys() == expected._arrays.keys(), index
        for key, array in storage._arrays.items():
            assert _same_array(array, expected._arrays[key]), (index, key)
        assert storage._shared == expected._shared, index


def _assert_index_invariants(storages, cleared):
    own_sets = []
    for index, storage in enumerate(storages):
        shared = storage._shared
        # The one shared-nothing object, exactly when nothing is shared, and
        # an attribute of the storage's own exactly when something is.
        assert (shared is _NOTHING_SHARED) == (not shared), index
        assert ("_shared" in vars(storage)) == bool(shared), index
        if shared is not _NOTHING_SHARED:
            own_sets.append(shared)
        if index not in cleared:
            assert shared <= storage._arrays.keys(), index
        for key in shared & storage._arrays.keys():
            # A shared array cannot be written through.
            assert not storage._arrays[key].flags.writeable, (index, key)
    # A set belongs to one storage.
    assert len({id(shared) for shared in own_sets}) == len(own_sets)
    assert not _NOTHING_SHARED


@hypothesis.settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(
    is_eternal=st.booleans(),
    operations=st.lists(_operation, max_size=60),
)
def test_storage_without_its_own_set_matches_storage_with_one(is_eternal, operations):
    tested = [InMemoryStorage(is_eternal)]
    reference = [_EagerIndexStorage(is_eternal)]
    cleared = set()

    for step, operation in enumerate(operations):
        read = _apply(tested, operation)
        expected = _apply(reference, operation)
        assert _same_array(read, expected), (step, operation)
        if operation[0] == "clear":
            cleared.add(operation[1] % len(tested))
        _assert_same_state(tested, reference)
        _assert_index_invariants(tested, cleared)

    # Every storage reads the same value for everything.
    for index, (storage, expected) in enumerate(zip(tested, reference)):
        for branch in BRANCHES:
            for period in PERIODS:
                assert _same_array(
                    storage.get(period, branch), expected.get(period, branch)
                ), (index, branch, period)
    _assert_same_state(tested, reference)
    _assert_index_invariants(tested, cleared)
