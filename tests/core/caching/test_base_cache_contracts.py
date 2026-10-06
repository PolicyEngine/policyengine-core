from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from policyengine_core.caching import (
    BaseCache,
    BoundedCache,
    BranchableCache,
    CacheClosedError,
    FactoryBackedCache,
    InvalidCacheKeyError,
    InvalidCacheValueError,
    RevisionAwareCache,
)


class DictCache(BaseCache[str, int]):
    def __init__(self) -> None:
        super().__init__()
        self.values: OrderedDict[str, int] = OrderedDict()

    def _lookup(self, key: str) -> int:
        return self.values[key]

    def _store(self, key: str, value: int) -> None:
        self.values[key] = value

    def _remove(self, key: str) -> bool:
        try:
            del self.values[key]
        except KeyError:
            return False
        return True

    def _clear_entries(self) -> int:
        count = len(self.values)
        self.values.clear()
        return count

    def _entry_count(self) -> int:
        return len(self.values)

    def _validate_key(self, key: str) -> None:
        if not isinstance(key, str) or not key:
            raise InvalidCacheKeyError(key)

    def _validate_value(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidCacheValueError(value)


class FactoryDictCache(FactoryBackedCache[str, int], DictCache):
    def __init__(self, enabled: bool = True) -> None:
        DictCache.__init__(self)
        self._enabled = enabled

    @property
    def cache_enabled(self) -> bool:
        return self._enabled


class BoundedDictCache(BoundedCache[str, int], DictCache):
    def __init__(self, max_entries: int) -> None:
        DictCache.__init__(self)
        self._max_entries = max_entries

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def _evict_one(self) -> bool:
        if not self.values:
            return False
        self.values.popitem(last=False)
        return True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("zero", 0, id="zero"),
        pytest.param("positive", 1, id="positive"),
        pytest.param("negative", -1, id="negative"),
        pytest.param("large", 10**12, id="large"),
        pytest.param("unicode", 7, id="unicode-key"),
        pytest.param("with space", 8, id="space-key"),
        pytest.param("a.b", 9, id="dotted-key"),
        pytest.param("2026-01-01", 10, id="date-like-key"),
    ],
)
def test_round_trip_supported_values(key: str, value: int) -> None:
    cache = DictCache()

    cache.put(key, value)

    assert cache.get(key) == value
    assert key in cache
    assert len(cache) == 1


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("absent", id="ordinary"),
        pytest.param("0", id="numeric-text"),
        pytest.param("not-present", id="hyphenated"),
        pytest.param("missing.period", id="dotted"),
    ],
)
def test_missing_key_raises_key_error(key: str) -> None:
    cache = DictCache()

    with pytest.raises(KeyError, match=key):
        cache.get(key)

    assert cache.cache_info().misses == 1


@pytest.mark.parametrize(
    ("key", "default"),
    [
        pytest.param("absent", 0, id="zero-default"),
        pytest.param("none", None, id="none-default"),
        pytest.param("negative", -1, id="negative-default"),
        pytest.param("large", 10**9, id="large-default"),
    ],
)
def test_missing_key_returns_explicit_default(key: str, default: Any) -> None:
    cache = DictCache()

    assert cache.get(key, default) is default
    assert cache.cache_info().misses == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(1, 2, id="increasing"),
        pytest.param(2, 1, id="decreasing"),
        pytest.param(-1, 0, id="negative-to-zero"),
        pytest.param(10**9, -(10**9), id="sign-change"),
    ],
)
def test_replacement_updates_value_without_growing(first: int, second: int) -> None:
    cache = DictCache()
    cache.put("key", first)

    cache.put("key", second)

    assert cache.get("key") == second
    assert len(cache) == 1
    assert cache.cache_info().writes == 2


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        pytest.param(True, True, id="present"),
        pytest.param(False, False, id="absent"),
    ],
)
def test_discard_reports_whether_entry_existed(present: bool, expected: bool) -> None:
    cache = DictCache()
    if present:
        cache.put("key", 1)

    assert cache.discard("key") is expected
    assert cache.cache_info().deletions == int(expected)


@pytest.mark.parametrize(
    "entry_count",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single"),
        pytest.param(3, id="several"),
        pytest.param(10, id="many"),
    ],
)
def test_clear_removes_all_entries(entry_count: int) -> None:
    cache = DictCache()
    for index in range(entry_count):
        cache.put(str(index), index)

    cache.clear()

    assert len(cache) == 0
    assert cache.cache_info().deletions == entry_count


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="mapping"),
        pytest.param(set(), id="set"),
    ],
)
def test_invalid_key_cannot_modify_cache(key: Any) -> None:
    cache = DictCache()
    cache.put("existing", 1)
    before = cache.cache_info()

    with pytest.raises(InvalidCacheKeyError):
        cache.put(key, 2)

    assert cache.get("existing") == 1
    assert cache.cache_info().writes == before.writes


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="mapping"),
        pytest.param(True, id="boolean-is-not-an-integer-value"),
    ],
)
def test_invalid_value_cannot_modify_cache(value: Any) -> None:
    cache = DictCache()
    cache.put("existing", 1)
    before = cache.cache_info()

    with pytest.raises(InvalidCacheValueError):
        cache.put("new", value)

    assert "new" not in cache
    assert cache.cache_info().writes == before.writes


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda cache: cache.get("key"), id="get"),
        pytest.param(lambda cache: cache.put("key", 2), id="put"),
        pytest.param(lambda cache: cache.discard("key"), id="discard"),
        pytest.param(lambda cache: cache.clear(), id="clear"),
        pytest.param(lambda cache: len(cache), id="length"),
        pytest.param(lambda cache: "key" in cache, id="contains"),
        pytest.param(lambda cache: cache["key"], id="getitem"),
    ],
)
def test_operations_after_close_raise(operation) -> None:
    cache = DictCache()
    cache.put("key", 1)
    cache.close()

    with pytest.raises(CacheClosedError):
        operation(cache)

    assert cache.cache_info().closed is True
    assert cache.cache_info().entries == 0


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        pytest.param(("miss",), (0, 1, 0, 0), id="one-miss"),
        pytest.param(("put",), (0, 0, 1, 1), id="one-write"),
        pytest.param(("put", "hit"), (1, 0, 1, 1), id="write-then-hit"),
        pytest.param(("put", "replace"), (0, 0, 2, 1), id="replacement"),
        pytest.param(("put", "delete"), (0, 0, 1, 0), id="deletion"),
        pytest.param(("put", "clear"), (0, 0, 1, 0), id="clear"),
    ],
)
def test_metrics_follow_documented_sequences(actions, expected) -> None:
    cache = DictCache()
    for action in actions:
        if action == "miss":
            cache.get("missing", None)
        elif action == "put":
            cache.put("key", 1)
        elif action == "hit":
            cache.get("key")
        elif action == "replace":
            cache.put("key", 2)
        elif action == "delete":
            cache.discard("key")
        elif action == "clear":
            cache.clear()

    info = cache.cache_info()

    assert (info.hits, info.misses, info.writes, info.entries) == expected


@pytest.mark.parametrize(
    "raises",
    [
        pytest.param(False, id="normal-exit"),
        pytest.param(True, id="exceptional-exit"),
    ],
)
def test_context_manager_always_closes(raises: bool) -> None:
    cache = DictCache()

    with pytest.raises(RuntimeError) if raises else _does_not_raise():
        with cache:
            cache.put("key", 1)
            if raises:
                raise RuntimeError("failure")

    assert cache.cache_info().closed is True


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("miss", id="construct-on-miss"),
        pytest.param("hit", id="reuse-on-hit"),
        pytest.param("disabled", id="construct-while-disabled"),
        pytest.param("factory-error", id="factory-error"),
        pytest.param("invalid-value", id="invalid-factory-value"),
    ],
)
def test_factory_behavior_is_atomic(scenario: str) -> None:
    cache = FactoryDictCache(enabled=scenario != "disabled")
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if scenario == "factory-error":
            raise RuntimeError("failure")
        if scenario == "invalid-value":
            return None
        return calls

    if scenario == "hit":
        cache.put("key", 99)
        assert cache.get_or_create("key", factory) == 99
        assert calls == 0
    elif scenario in {"factory-error", "invalid-value"}:
        error = RuntimeError if scenario == "factory-error" else InvalidCacheValueError
        with pytest.raises(error):
            cache.get_or_create("key", factory)
        assert len(cache) == 0
    else:
        first = cache.get_or_create("key", factory)
        second = cache.get_or_create("key", factory)
        assert first == 1
        if scenario == "disabled":
            assert second == 2
            assert len(cache) == 0
        else:
            assert second == 1
            assert len(cache) == 1


@pytest.mark.parametrize(
    "max_entries",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(2, id="two"),
        pytest.param(3, id="three"),
    ],
)
def test_bounded_cache_never_exceeds_capacity(max_entries: int) -> None:
    cache = BoundedDictCache(max_entries)

    for index in range(6):
        cache.put(str(index), index)

    assert len(cache) == max_entries
    assert cache.cache_info().evictions == 6 - max_entries
    assert list(cache.values) == [str(index) for index in range(6 - max_entries, 6)]


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("put", id="concurrent-writes"),
        pytest.param("get", id="concurrent-reads"),
        pytest.param("factory", id="single-construction"),
    ],
)
def test_public_operations_are_thread_safe(operation: str) -> None:
    cache = FactoryDictCache()
    cache.put("shared", 1)
    calls = 0
    calls_lock = threading.Lock()

    def act(index: int) -> int:
        nonlocal calls
        if operation == "put":
            cache.put(f"key-{index}", index)
            return index
        if operation == "get":
            return cache.get("shared")

        def factory() -> int:
            nonlocal calls
            with calls_lock:
                calls += 1
            return 7

        return cache.get_or_create("constructed", factory)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(act, range(32)))

    assert len(results) == 32
    if operation == "put":
        assert len(cache) == 33
    elif operation == "get":
        assert results == [1] * 32
    else:
        assert results == [7] * 32
        assert calls == 1


@pytest.mark.parametrize(
    "abstract_class",
    [
        pytest.param(BaseCache, id="base"),
        pytest.param(RevisionAwareCache, id="revision-aware"),
        pytest.param(BranchableCache, id="branchable"),
    ],
)
def test_incomplete_abstract_cache_cannot_be_instantiated(abstract_class) -> None:
    with pytest.raises(TypeError):
        abstract_class()


class _does_not_raise:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False
