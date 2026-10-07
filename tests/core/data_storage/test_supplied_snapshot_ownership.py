"""Unit contracts for storage-owned supplied snapshots (no country model)."""

import copy

import numpy as np
import pytest

from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)
from policyengine_core.enums import Enum
from policyengine_core.periods import period


class Choice(Enum):
    first = "first"
    second = "second"


@pytest.fixture(params=["memory", "disk"], ids=["memory", "disk"])
def storage(request, tmp_path):
    if request.param == "memory":
        return InMemoryStorage(False)
    return OnDiskStorage(str(tmp_path), preserve_storage_dir=True)


def test_supplied_value_survives_retention(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    storage.retain_supplied_inputs()
    np.testing.assert_array_equal(storage.get("2025-01"), [4])
    assert storage.is_supplied("2025-01")


def test_derived_value_is_removed(storage):
    storage.put(np.array([4]), "2025-01", derived=True)
    storage.retain_supplied_inputs()
    assert not storage.has("2025-01")


def test_legacy_nonderived_value_is_not_supplied(storage):
    storage.put(np.array([4]), "2025-01")
    assert not storage.is_supplied("2025-01")
    storage.retain_supplied_inputs()
    assert not storage.has("2025-01")


def test_source_mutation_cannot_change_supplied_value(storage):
    source = np.array([4])
    storage.put(source, "2025-01", supplied=True)
    source[0] = 9
    np.testing.assert_array_equal(storage.get("2025-01"), [4])


def test_retention_keeps_exact_payload_without_read_or_write(storage, monkeypatch):
    storage.put(np.array([4]), "2025-01", supplied=True)

    def fail(*args, **kwargs):
        raise AssertionError("retention must not read or write supplied arrays")

    monkeypatch.setattr(storage, "get", fail)
    monkeypatch.setattr(storage, "put", fail)
    storage.retain_supplied_inputs()
    assert storage.has("2025-01")


def test_replacing_supplied_with_cache_removes_provenance(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    storage.put(np.array([5]), "2025-01", derived=True)
    assert not storage.is_supplied("2025-01")


def test_replacing_cache_with_supplied_retains_provenance(storage):
    storage.put(np.array([4]), "2025-01", derived=True)
    storage.put(np.array([5]), "2025-01", supplied=True)
    assert storage.is_supplied("2025-01")
    assert not storage.is_derived("2025-01")


def test_clone_has_independent_supplied_metadata(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    clone = storage.clone()
    clone.delete("2025-01")
    assert storage.is_supplied("2025-01")
    assert not clone.is_supplied("2025-01")


def test_clone_replacement_does_not_mutate_source(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    clone = storage.clone()
    clone.put(np.array([8]), "2025-01", supplied=True)
    np.testing.assert_array_equal(storage.get("2025-01"), [4])
    np.testing.assert_array_equal(clone.get("2025-01"), [8])


def test_delete_year_removes_months_and_provenance(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    storage.put(np.array([5]), "2026-01", supplied=True)
    storage.delete("2025")
    assert not storage.has("2025-01")
    assert not storage.is_supplied("2025-01")
    assert storage.is_supplied("2026-01")


def test_underscore_branch_remains_distinct(storage):
    storage.put(np.array([4]), "2025-01", "spm_branch", supplied=True)
    storage.put(np.array([5]), "2025-01", "spm", supplied=True)
    storage.delete(branch_name="spm")
    assert storage.is_supplied("2025-01", "spm_branch")
    assert not storage.is_supplied("2025-01", "spm")
    assert storage.get_known_branch_periods() == [("spm_branch", period("2025-01"))]


def test_eternity_input_metadata_uses_canonical_period(storage):
    storage.is_eternal = True
    storage.put(np.array([4]), "2025-01", supplied=True)
    assert storage.is_supplied("2026")
    storage.delete("2027")
    assert not storage.is_supplied("2025-01")


def test_copy_keeps_supplied_metadata(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    clone = copy.deepcopy(storage)
    clone.delete("2025-01")
    assert storage.is_supplied("2025-01")
    assert not clone.is_supplied("2025-01")


def test_contradictory_provenance_is_rejected_without_mutation(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    with pytest.raises(ValueError):
        storage.put(np.array([8]), "2025-01", derived=True, supplied=True)
    np.testing.assert_array_equal(storage.get("2025-01"), [4])
    assert storage.is_supplied("2025-01")


def test_retention_removes_only_this_storage_results(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    storage.put(np.array([5]), "2025-02", derived=True)
    clone = storage.clone()
    storage.retain_supplied_inputs()
    assert clone.has("2025-02")
    assert not storage.has("2025-02")


def test_empty_retention_is_idempotent(storage):
    storage.retain_supplied_inputs()
    storage.retain_supplied_inputs()
    assert storage.get_known_periods() == []


ARRAY_FACTORIES = [
    pytest.param(lambda: np.array([1.0, 2.0]), id="float-array"),
    pytest.param(lambda: np.array([1, 2]), id="integer-array"),
    pytest.param(lambda: np.array([True, False]), id="boolean-array"),
    pytest.param(lambda: np.arange(6)[::2], id="strided-view"),
    pytest.param(lambda: np.ma.array([1, 2], mask=[False, True]), id="masked-array"),
    pytest.param(lambda: Choice.encode(np.array(["first", "second"])), id="enum-array"),
]


@pytest.mark.parametrize("factory", ARRAY_FACTORIES)
def test_direct_entry_constructor_owns_its_source(factory):
    source = factory()
    expected = source.copy()
    entry = CachedArrayEntry(source)
    source[...] = source[::-1]
    np.testing.assert_array_equal(entry.read(), expected)
    assert not np.shares_memory(source, entry.read())


@pytest.mark.parametrize("factory", ARRAY_FACTORIES)
def test_direct_entry_cannot_be_made_writable(factory):
    entry = CachedArrayEntry(factory())
    with pytest.raises(ValueError):
        entry.read().flags.writeable = True


@pytest.mark.parametrize("factory", ARRAY_FACTORIES)
def test_cache_put_of_direct_entry_cannot_bypass_snapshot(factory):
    source = factory()
    expected = source.copy()
    entry = CachedArrayEntry(source)
    cache = ImmutableArrayCache()
    cache.put("key", entry)
    source[...] = source[::-1]
    np.testing.assert_array_equal(cache.get_array("key"), expected)


def test_direct_masked_entry_copies_mask():
    source = np.ma.array([1, 2], mask=[False, True])
    entry = CachedArrayEntry(source)
    source.mask[:] = False
    np.testing.assert_array_equal(entry.read().mask, [False, True])


def test_retention_preserves_memory_entry_identity():
    storage = InMemoryStorage(False)
    entry = storage.put(np.array([4]), "2025-01", supplied=True)
    storage.put(np.array([9]), "2025-02", derived=True)
    storage.retain_supplied_inputs()
    assert storage._entry_cache.entry("default:2025-01") is entry


def test_exact_discard_keeps_contained_input(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    storage.put(np.array([9]), "2025", supplied=True)
    storage.discard("2025")
    assert not storage.is_supplied("2025")
    assert storage.is_supplied("2025-01")
    np.testing.assert_array_equal(storage.get("2025-01"), [4])


def test_exact_discard_does_not_change_clone(storage):
    storage.put(np.array([4]), "2025-01", supplied=True)
    clone = storage.clone()
    clone.discard("2025-01")
    assert storage.is_supplied("2025-01")
    assert not clone.is_supplied("2025-01")
