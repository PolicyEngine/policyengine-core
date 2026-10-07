"""Compatibility and lifecycle tests for immutable in-memory entries."""

from __future__ import annotations

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from policyengine_core.data_storage import InMemoryStorage
from policyengine_core.data_storage.in_memory_storage import _NOTHING_SHARED


def _storage_with(*input_periods: str) -> InMemoryStorage:
    storage = InMemoryStorage(is_eternal=False)
    for index, input_period in enumerate(input_periods):
        storage.put(np.array([index, index + 1.0]), input_period)
    return storage


def test_immutable_storage_never_allocates_a_shared_key_index() -> None:
    source = _storage_with("2017-01", "2017-02")
    clone = source.clone(share_arrays=True)
    assert source._shared is _NOTHING_SHARED
    assert clone._shared is _NOTHING_SHARED
    assert "_shared" not in vars(source)
    assert "_shared" not in vars(clone)


@pytest.mark.parametrize("share_arrays", [False, True], ids=["legacy-copy", "share"])
def test_clone_modes_share_immutable_entries(share_arrays: bool) -> None:
    source = _storage_with("2017-01", "2017-02")
    clone = source.clone(share_arrays=share_arrays)
    assert clone._arrays.keys() == source._arrays.keys()
    for key, entry in source._arrays.items():
        assert clone._arrays[key] is entry


def test_reading_does_not_replace_a_shared_entry() -> None:
    source = _storage_with("2017-01")
    clone = source.clone(share_arrays=True)
    entry = clone._arrays["default:2017-01"]
    first = clone.get("2017-01")
    second = clone.get("2017-01")
    assert clone._arrays["default:2017-01"] is entry
    assert first is not second
    assert np.shares_memory(first, second)
    assert not first.flags.writeable and not second.flags.writeable


def test_clone_replacement_does_not_change_source_index() -> None:
    source = _storage_with("2017-01")
    clone = source.clone(share_arrays=True)
    clone.put(np.array([9.0, 10.0]), "2017-01")
    np.testing.assert_array_equal(source.get("2017-01"), [0.0, 1.0])
    np.testing.assert_array_equal(clone.get("2017-01"), [9.0, 10.0])
    assert clone._arrays["default:2017-01"] is not source._arrays["default:2017-01"]


def test_clone_deletion_does_not_change_source_index() -> None:
    source = _storage_with("2017-01", "2017-02")
    clone = source.clone(share_arrays=True)
    clone.delete("2017-01")
    assert clone.get("2017-01") is None
    np.testing.assert_array_equal(source.get("2017-01"), [0.0, 1.0])
    np.testing.assert_array_equal(clone.get("2017-02"), [1.0, 2.0])


def test_external_legacy_clear_changes_only_that_index() -> None:
    source = _storage_with("2017-01")
    clone = source.clone(share_arrays=True)
    clone._arrays.clear()
    assert clone.get("2017-01") is None
    np.testing.assert_array_equal(source.get("2017-01"), [0.0, 1.0])


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["deepcopy", "pickle"],
)
def test_duplicate_keeps_values_and_has_an_independent_index(duplicate) -> None:
    source = _storage_with("2017-01", "2017-02")
    twin = duplicate(source)
    twin.delete("2017-01")
    assert twin.get("2017-01") is None
    np.testing.assert_array_equal(source.get("2017-01"), [0.0, 1.0])
    assert twin._shared is _NOTHING_SHARED


def test_concurrent_reads_return_protected_equivalent_views() -> None:
    storage = _storage_with("2017-01")
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(lambda _: storage.get("2017-01"), range(100)))
    assert all(np.array_equal(value, [0.0, 1.0]) for value in values)
    assert all(not value.flags.writeable for value in values)
    assert len({id(value) for value in values}) == len(values)


def test_nested_clones_share_one_entry_with_independent_indexes() -> None:
    source = _storage_with("2017-01")
    child = source.clone(share_arrays=True)
    grandchild = child.clone(share_arrays=True)
    key = "default:2017-01"
    assert source._arrays[key] is child._arrays[key] is grandchild._arrays[key]
    child.delete("2017-01")
    assert (
        key in source._arrays and key not in child._arrays and key in grandchild._arrays
    )
