"""Property test for immutable value sharing with independent indexes."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.data_storage import InMemoryStorage

PERIODS = ("2017-01", "2017-02", "2017")
BRANCHES = ("default", "other")

operation = st.one_of(
    st.tuples(
        st.just("put"),
        st.integers(min_value=0, max_value=10**6),
        st.sampled_from(PERIODS),
        st.sampled_from(BRANCHES),
        st.integers(min_value=-100, max_value=100),
        st.booleans(),
    ),
    st.tuples(
        st.just("get"),
        st.integers(min_value=0, max_value=10**6),
        st.sampled_from(PERIODS),
        st.sampled_from(BRANCHES),
    ),
    st.tuples(
        st.just("delete"),
        st.integers(min_value=0, max_value=10**6),
        st.none() | st.sampled_from(PERIODS),
        st.sampled_from(BRANCHES),
    ),
    st.tuples(
        st.just("clone"),
        st.integers(min_value=0, max_value=10**6),
        st.booleans(),
    ),
    st.tuples(st.just("clear"), st.integers(min_value=0, max_value=10**6)),
)


def _snapshot(storage: InMemoryStorage) -> dict[str, tuple[np.ndarray, bool]]:
    return {
        key: (entry.read().copy(), key not in storage._inputs)
        for key, entry in storage.entry_cache.entries.items()
    }


def _assert_snapshot(storage: InMemoryStorage, expected) -> None:
    assert storage.entry_cache.entries.keys() == expected.keys()
    for key, (value, derived) in expected.items():
        branch, input_period = key.split(":", 1)
        result = storage.get(input_period, branch)
        np.testing.assert_array_equal(result, value)
        assert not result.flags.writeable
        assert storage.is_derived(input_period, branch) is derived


@hypothesis.settings(max_examples=500, deadline=None)
@hypothesis.given(
    is_eternal=st.booleans(),
    operations=st.lists(operation, max_size=80),
)
def test_random_operations_preserve_snapshots_and_index_isolation(
    is_eternal,
    operations,
) -> None:
    storages = [InMemoryStorage(is_eternal)]
    snapshots = [_snapshot(storages[0])]

    for item in operations:
        kind = item[0]
        index = item[1] % len(storages)
        storage = storages[index]
        if kind == "put":
            _, _, input_period, branch, seed, derived = item
            source = np.array([seed, seed + 1], dtype=np.float64)
            storage.put(source, input_period, branch, derived=derived)
            source[:] = 999
            snapshots[index] = _snapshot(storage)
        elif kind == "get":
            _, _, input_period, branch = item
            result = storage.get(input_period, branch)
            if result is not None:
                assert not result.flags.writeable
        elif kind == "delete":
            _, _, input_period, branch = item
            storage.delete(input_period, branch)
            snapshots[index] = _snapshot(storage)
        elif kind == "clone":
            clone = storage.clone(share_arrays=item[2])
            storages.append(clone)
            snapshots.append(_snapshot(clone))
        else:
            for branch, _ in storage.get_known_branch_periods():
                storage.delete(branch_name=branch)
            snapshots[index] = {}

        for candidate, expected in zip(storages, snapshots):
            _assert_snapshot(candidate, expected)
