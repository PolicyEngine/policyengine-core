from __future__ import annotations

import numpy as np
import pytest

from policyengine_core.caching import (
    CacheClosedError,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)
from policyengine_core.data_storage import InMemoryStorage, OnDiskStorage
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)


ENTRY_CASES = [
    ("int8", lambda: np.array([-1, 0, 1], dtype=np.int8)),
    ("int16", lambda: np.array([-2, 0, 2], dtype=np.int16)),
    ("int32", lambda: np.array([-3, 0, 3], dtype=np.int32)),
    ("int64", lambda: np.array([-4, 0, 4], dtype=np.int64)),
    ("uint8", lambda: np.array([0, 1, 255], dtype=np.uint8)),
    ("float32", lambda: np.array([1.5, np.nan], dtype=np.float32)),
    ("float64", lambda: np.array([2.5, np.inf], dtype=np.float64)),
    ("bool", lambda: np.array([True, False], dtype=bool)),
    ("unicode", lambda: np.array(["one", "two"])),
    ("bytes", lambda: np.array([b"one", b"two"])),
    ("datetime", lambda: np.array(["2024-01-01"], dtype="datetime64[D]")),
    ("timedelta", lambda: np.array([1, 2], dtype="timedelta64[D]")),
    ("empty", lambda: np.array([], dtype=np.float32)),
    ("matrix", lambda: np.arange(6).reshape(2, 3)),
    ("cube", lambda: np.arange(8).reshape(2, 2, 2)),
    ("noncontiguous", lambda: np.arange(12).reshape(3, 4)[:, ::2]),
]


@pytest.mark.parametrize(
    "_name,make", ENTRY_CASES, ids=[case[0] for case in ENTRY_CASES]
)
def test_cached_array_entry_returns_an_immutable_equivalent_array(
    _name: str,
    make,
) -> None:
    source = make()
    entry = CachedArrayEntry.from_value(source)
    value = entry.read()
    np.testing.assert_equal(value, source)
    assert value.dtype == source.dtype
    assert value.shape == source.shape
    assert value.flags.writeable is False
    with pytest.raises(ValueError):
        value.flags.writeable = True


def test_cached_array_entry_keeps_its_owned_snapshot_read_only() -> None:
    entry = CachedArrayEntry.from_value(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        entry._value.flags.writeable = True


def test_cached_masked_array_protects_data_and_mask_buffers() -> None:
    entry = CachedArrayEntry.from_value(np.ma.array([1.0, 2.0], mask=[False, True]))
    value = entry.read()

    with pytest.raises(ValueError):
        value[0] = 3.0
    with pytest.raises(ValueError):
        value.mask[0] = True
    with pytest.raises(ValueError):
        value.flags.writeable = True
    with pytest.raises(ValueError):
        value.mask.flags.writeable = True


@pytest.mark.parametrize(
    "mutation",
    [
        "set-first",
        "set-last",
        "fill",
        "add",
        "multiply",
        "sort",
        "reverse",
        "reshape",
        "change-dtype",
        "resize-larger",
        "resize-smaller",
        "set-all",
    ],
)
def test_cached_entry_does_not_follow_later_source_mutation(mutation: str) -> None:
    source = np.arange(6, dtype=np.float64)
    expected = source.copy()
    entry = CachedArrayEntry.from_value(source)
    if mutation == "set-first":
        source[0] = 99
    elif mutation == "set-last":
        source[-1] = 99
    elif mutation == "fill":
        source.fill(99)
    elif mutation == "add":
        source += 10
    elif mutation == "multiply":
        source *= 10
    elif mutation == "sort":
        source[:] = source[::-1]
        source.sort()
    elif mutation == "reverse":
        source[:] = source[::-1]
    elif mutation == "reshape":
        source.shape = (2, 3)
    elif mutation == "change-dtype":
        source.dtype = np.int64
    elif mutation == "resize-larger":
        source.resize(10)
    elif mutation == "resize-smaller":
        source.resize(3)
    else:
        source[:] = -1
    np.testing.assert_array_equal(entry.read(), expected)
    assert entry.read().shape == expected.shape
    assert entry.read().dtype == expected.dtype


@pytest.mark.parametrize(
    "operation,index",
    [
        ("put-get", 0),
        ("put-get", 1),
        ("replace", 2),
        ("replace", 3),
        ("discard", 4),
        ("discard", 5),
        ("discard-missing", 6),
        ("discard-missing", 7),
        ("clear", 8),
        ("clear", 9),
        ("missing", 10),
        ("missing", 11),
        ("invalid-key", 12),
        ("invalid-value", 13),
        ("closed-read", 14),
        ("closed-write", 15),
    ],
    ids=lambda value: str(value),
)
def test_immutable_array_cache_contract(operation: str, index: int) -> None:
    cache: ImmutableArrayCache[str, np.ndarray] = ImmutableArrayCache()
    key = f"key-{index}"
    value = np.array([index], dtype=np.int64)
    if operation == "put-get":
        cache.put_array(key, value)
        np.testing.assert_array_equal(cache.get_array(key), value)
    elif operation == "replace":
        cache.put_array(key, value)
        cache.put_array(key, value + 1)
        np.testing.assert_array_equal(cache.get_array(key), value + 1)
    elif operation == "discard":
        cache.put_array(key, value)
        assert cache.discard(key) is True
        assert cache.get_array(key, None) is None
    elif operation == "discard-missing":
        assert cache.discard(key) is False
    elif operation == "clear":
        cache.put_array(key, value)
        cache.put_array(f"other-{index}", value)
        cache.clear()
        assert len(cache) == 0
    elif operation == "missing":
        with pytest.raises(KeyError):
            cache.get_array(key)
    elif operation == "invalid-key":
        with pytest.raises(InvalidCacheKeyError):
            cache.put_array([], value)  # type: ignore[arg-type]
    elif operation == "invalid-value":
        with pytest.raises(InvalidCacheValueError):
            cache.put(key, value)  # type: ignore[arg-type]
    elif operation == "closed-read":
        cache.close()
        with pytest.raises(CacheClosedError):
            cache.get_array(key)
    else:
        cache.close()
        with pytest.raises(CacheClosedError):
            cache.put_array(key, value)


@pytest.mark.parametrize(
    "operation,index",
    [
        ("shared-entry", 0),
        ("shared-entry", 1),
        ("independent-index", 2),
        ("independent-index", 3),
        ("source-replace", 4),
        ("source-replace", 5),
        ("fork-replace", 6),
        ("fork-replace", 7),
        ("source-delete", 8),
        ("source-delete", 9),
        ("fork-delete", 10),
        ("fork-delete", 11),
        ("nested", 12),
        ("nested", 13),
        ("read-only", 14),
        ("read-only", 15),
    ],
    ids=lambda value: str(value),
)
def test_immutable_array_cache_forks_share_values_not_indexes(
    operation: str,
    index: int,
) -> None:
    cache: ImmutableArrayCache[str, np.ndarray] = ImmutableArrayCache()
    key = f"key-{index}"
    original = np.array([index, index + 1], dtype=np.float64)
    cache.put_array(key, original)
    fork = cache.fork()
    if operation == "shared-entry":
        assert cache.entry(key) is fork.entry(key)
    elif operation == "independent-index":
        fork.put_array("fork-only", original)
        assert "fork-only" not in cache and "fork-only" in fork
    elif operation == "source-replace":
        cache.put_array(key, original + 10)
        np.testing.assert_array_equal(fork.get_array(key), original)
    elif operation == "fork-replace":
        fork.put_array(key, original + 10)
        np.testing.assert_array_equal(cache.get_array(key), original)
    elif operation == "source-delete":
        cache.discard(key)
        np.testing.assert_array_equal(fork.get_array(key), original)
    elif operation == "fork-delete":
        fork.discard(key)
        np.testing.assert_array_equal(cache.get_array(key), original)
    elif operation == "nested":
        grandchild = fork.fork()
        assert grandchild.entry(key) is cache.entry(key)
        fork.discard(key)
        np.testing.assert_array_equal(grandchild.get_array(key), original)
    else:
        for value in (cache.get_array(key), fork.get_array(key)):
            with pytest.raises(ValueError):
                value[0] = -1


@pytest.mark.parametrize(
    "branch_name,input_period,is_eternal,derived",
    [
        ("default", "2024-01", False, False),
        ("default", "2024-01", False, True),
        ("reform", "2024-01", False, False),
        ("reform", "2024-01", False, True),
        ("nested", "2024", False, False),
        ("nested", "2024", False, True),
        ("default", "ETERNITY", True, False),
        ("default", "2024", True, True),
        ("reform", "ETERNITY", True, False),
        ("reform", "2024-01", True, True),
        ("branch-a", "2024-12", False, False),
        ("branch-b", "2024-12", False, True),
    ],
)
def test_in_memory_storage_returns_read_only_values_and_preserves_marks(
    branch_name: str,
    input_period: str,
    is_eternal: bool,
    derived: bool,
) -> None:
    storage = InMemoryStorage(is_eternal=is_eternal)
    source = np.array([1.0, 2.0])
    storage.put(source, input_period, branch_name, derived=derived)
    value = storage.get(input_period, branch_name)
    np.testing.assert_array_equal(value, [1.0, 2.0])
    assert value.flags.writeable is False
    assert storage.is_derived(input_period, branch_name) is derived
    source[0] = 99
    assert storage.get(input_period, branch_name)[0] == 1.0


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.int8, [1, 2]),
        (np.int16, [1, 2]),
        (np.int32, [1, 2]),
        (np.int64, [1, 2]),
        (np.float32, [1.5, 2.5]),
        (np.float64, [1.5, 2.5]),
        (np.bool_, [True, False]),
        (np.str_, ["one", "two"]),
    ],
    ids=["int8", "int16", "int32", "int64", "float32", "float64", "bool", "str"],
)
def test_on_disk_storage_returns_read_only_arrays(tmp_path, dtype, values) -> None:
    storage = OnDiskStorage(str(tmp_path), preserve_storage_dir=True)
    storage.put(np.array(values, dtype=dtype), "2024")
    result = storage.get("2024")
    assert result.flags.writeable is False
    with pytest.raises(ValueError):
        result[0] = result[-1]
