from __future__ import annotations

import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest

from policyengine_core.caching import (
    CacheClosedError,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)
from policyengine_core.tools.policy_system_cache import (
    PolicySystemCache,
    PolicySystemCacheKey,
    freeze_cache_value,
)


@dataclass
class FakeSystem:
    identity: int


BASELINE_KEY = PolicySystemCacheKey.from_inputs()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, ("none", None), id="none"),
        pytest.param(True, ("bool", True), id="true"),
        pytest.param(False, ("bool", False), id="false"),
        pytest.param(0, ("int", 0), id="zero"),
        pytest.param(-4, ("int", -4), id="negative-integer"),
        pytest.param(1.5, ("float", 1.5), id="float"),
        pytest.param("", ("str", ""), id="empty-string"),
        pytest.param("rate", ("str", "rate"), id="string"),
        pytest.param([1, 2], ("list", (("int", 1), ("int", 2))), id="list"),
        pytest.param((1, "x"), ("tuple", (("int", 1), ("str", "x"))), id="tuple"),
        pytest.param(
            {"b": 2, "a": 1},
            (
                "mapping",
                (("a", ("int", 1)), ("b", ("int", 2))),
            ),
            id="mapping",
        ),
        pytest.param(
            datetime.date(2026, 1, 2),
            ("date", "2026-01-02"),
            id="date",
        ),
    ],
)
def test_freeze_cache_value_preserves_type_and_content(value: Any, expected) -> None:
    assert freeze_cache_value(value) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param({"a": 1, "b": 2}, {"b": 2, "a": 1}, id="mapping-order"),
        pytest.param({"a": [1, 2]}, {"a": [1, 2]}, id="nested-list"),
        pytest.param({"a": {"b": True}}, {"a": {"b": True}}, id="nested-map"),
        pytest.param([{"a": 1}], [{"a": 1}], id="mapping-in-list"),
        pytest.param((1, [2]), (1, [2]), id="list-in-tuple"),
        pytest.param(
            datetime.date(2025, 12, 31), datetime.date(2025, 12, 31), id="date"
        ),
        pytest.param("é", "é", id="unicode"),
        pytest.param(-0.0, -0.0, id="negative-zero"),
    ],
)
def test_equivalent_values_have_equal_frozen_keys(left: Any, right: Any) -> None:
    assert freeze_cache_value(left) == freeze_cache_value(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param(True, 1, id="boolean-versus-integer"),
        pytest.param(1, 1.0, id="integer-versus-float"),
        pytest.param("1", 1, id="string-versus-integer"),
        pytest.param([], (), id="list-versus-tuple"),
        pytest.param(None, "", id="none-versus-empty-string"),
        pytest.param({"a": 1}, {"a": "1"}, id="nested-type"),
        pytest.param([1, 2], [2, 1], id="list-order"),
        pytest.param(datetime.date(2026, 1, 1), "2026-01-01", id="date-versus-string"),
    ],
)
def test_distinct_typed_values_have_distinct_frozen_keys(left: Any, right: Any) -> None:
    assert freeze_cache_value(left) != freeze_cache_value(right)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param({1: "value"}, id="non-string-mapping-key"),
        pytest.param({1, 2}, id="set"),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_unsupported_key_value_is_rejected(value: Any) -> None:
    with pytest.raises(InvalidCacheKeyError):
        freeze_cache_value(value)


def example_reform(system):
    return system


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param({}, ((), (), ()), id="empty"),
        pytest.param(
            {"reforms": "package.reform"},
            (("path:package.reform",), (), ()),
            id="single-reform-string",
        ),
        pytest.param(
            {"reforms": ["a", "b"]},
            (("path:a", "path:b"), (), ()),
            id="ordered-reforms",
        ),
        pytest.param(
            {"extensions": "package.extension"},
            ((), ("package.extension",), ()),
            id="single-extension-string",
        ),
        pytest.param(
            {"extensions": ["a", "b"]},
            ((), ("a", "b"), ()),
            id="ordered-extensions",
        ),
        pytest.param(
            {"parameter_overrides": {"gov.rate": 0.2}},
            ((), (), (("gov.rate", ("float", 0.2)),)),
            id="one-override",
        ),
        pytest.param(
            {"parameter_overrides": {"b": 2, "a": 1}},
            ((), (), (("a", ("int", 1)), ("b", ("int", 2)))),
            id="sorted-overrides",
        ),
        pytest.param(
            {"reforms": [example_reform]},
            ((f"callable:{__name__}.example_reform",), (), ()),
            id="callable-reform",
        ),
    ],
)
def test_policy_system_key_normalizes_inputs(inputs: dict, expected) -> None:
    key = PolicySystemCacheKey.from_inputs(**inputs)

    assert (key.reforms, key.extensions, key.parameter_overrides) == expected


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("baseline-reuse", id="baseline-reuse"),
        pytest.param("derived-reuse", id="derived-reuse"),
        pytest.param("baseline-and-derived", id="baseline-and-derived"),
        pytest.param("capacity-zero-baseline", id="capacity-zero-baseline"),
        pytest.param("capacity-zero-derived", id="capacity-zero-derived"),
        pytest.param("metrics", id="metrics"),
        pytest.param("clear", id="clear"),
        pytest.param("close", id="close"),
    ],
)
def test_policy_system_cache_standard_behavior(scenario: str) -> None:
    capacity = 0 if scenario.startswith("capacity-zero") else 2
    cache = PolicySystemCache[FakeSystem](max_derived_entries=capacity)
    derived_key = PolicySystemCacheKey.from_inputs(reforms=["reform"])
    calls = 0

    def build() -> FakeSystem:
        nonlocal calls
        calls += 1
        return FakeSystem(calls)

    key = BASELINE_KEY if "baseline" in scenario else derived_key
    if scenario in {"baseline-reuse", "derived-reuse"}:
        assert cache.get_or_create(key, build) is cache.get_or_create(key, build)
        assert calls == 1
    elif scenario == "baseline-and-derived":
        cache.get_or_create(BASELINE_KEY, build)
        cache.get_or_create(derived_key, build)
        assert cache.cache_info().entries == 2
    elif scenario == "capacity-zero-baseline":
        first = cache.get_or_create(BASELINE_KEY, build)
        assert cache.get_or_create(BASELINE_KEY, build) is first
    elif scenario == "capacity-zero-derived":
        first = cache.get_or_create(derived_key, build)
        second = cache.get_or_create(derived_key, build)
        assert first is not second
        assert derived_key not in cache
    elif scenario == "metrics":
        cache.get_or_create(key, build)
        cache.get_or_create(key, build)
        info = cache.cache_info()
        assert (info.misses, info.hits, info.builds, info.writes) == (1, 1, 1, 1)
    elif scenario == "clear":
        cache.get_or_create(key, build)
        cache.clear()
        assert len(cache) == 0
    else:
        cache.get_or_create(key, build)
        cache.close()
        with pytest.raises(CacheClosedError):
            cache.get_or_create(key, build)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("baseline", id="baseline"),
        pytest.param("derived", id="derived"),
        pytest.param("repeated", id="repeated"),
        pytest.param("metrics", id="metrics"),
    ],
)
def test_disabled_policy_system_cache_never_retains(scenario: str) -> None:
    cache = PolicySystemCache[FakeSystem](enabled=False)
    key = (
        BASELINE_KEY
        if scenario == "baseline"
        else PolicySystemCacheKey.from_inputs(reforms=["r"])
    )
    calls = 0

    def build() -> FakeSystem:
        nonlocal calls
        calls += 1
        return FakeSystem(calls)

    first = cache.get_or_create(key, build)
    second = cache.get_or_create(key, build)

    assert first is not second
    assert len(cache) == 0
    assert calls == 2
    if scenario == "metrics":
        info = cache.cache_info()
        assert (info.misses, info.builds, info.writes, info.entries) == (2, 2, 0, 0)


@pytest.mark.parametrize(
    ("capacity", "accessed", "remaining"),
    [
        pytest.param(1, (), ("c",), id="capacity-one"),
        pytest.param(2, (), ("b", "c"), id="capacity-two"),
        pytest.param(2, ("a",), ("a", "c"), id="recently-used-survives"),
        pytest.param(3, (), ("a", "b", "c"), id="capacity-three"),
    ],
)
def test_derived_entries_use_lru_eviction(
    capacity: int, accessed: tuple, remaining: tuple
) -> None:
    cache = PolicySystemCache[FakeSystem](max_derived_entries=capacity)
    cache.get_or_create(BASELINE_KEY, lambda: FakeSystem(0))
    keys = {
        label: PolicySystemCacheKey.from_inputs(reforms=[label])
        for label in ("a", "b", "c")
    }
    for index, label in enumerate(("a", "b"), start=1):
        cache.get_or_create(keys[label], lambda index=index: FakeSystem(index))
    for label in accessed:
        cache.get(keys[label])
    cache.get_or_create(keys["c"], lambda: FakeSystem(3))

    assert tuple(label for label, key in keys.items() if key in cache) == remaining
    assert BASELINE_KEY in cache


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("none", id="none-value"),
        pytest.param("factory-error", id="factory-error"),
        pytest.param("invalid-key", id="invalid-key"),
    ],
)
def test_policy_system_cache_rejects_invalid_operations_atomically(
    scenario: str,
) -> None:
    cache = PolicySystemCache[FakeSystem]()
    cache.get_or_create(BASELINE_KEY, lambda: FakeSystem(0))
    before = cache.cache_info()

    if scenario == "none":
        with pytest.raises(InvalidCacheValueError):
            cache.get_or_create(
                PolicySystemCacheKey.from_inputs(reforms=["x"]), lambda: None
            )
    elif scenario == "factory-error":

        def fail():
            raise RuntimeError("failure")

        with pytest.raises(RuntimeError):
            cache.get_or_create(PolicySystemCacheKey.from_inputs(reforms=["x"]), fail)
    else:
        with pytest.raises(InvalidCacheKeyError):
            cache.get_or_create("not-a-key", lambda: FakeSystem(1))

    after = cache.cache_info()
    assert after.entries == before.entries
    assert after.writes == before.writes


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("same-key", id="same-key-single-build"),
        pytest.param("different-keys", id="different-keys"),
        pytest.param("disabled", id="disabled-builds-every-time"),
    ],
)
def test_policy_system_cache_concurrent_construction(scenario: str) -> None:
    cache = PolicySystemCache[FakeSystem](
        enabled=scenario != "disabled", max_derived_entries=32
    )
    calls = 0
    calls_lock = threading.Lock()

    def get(index: int) -> FakeSystem:
        key = PolicySystemCacheKey.from_inputs(
            reforms=[str(index)] if scenario == "different-keys" else ["shared"]
        )

        def build() -> FakeSystem:
            nonlocal calls
            with calls_lock:
                calls += 1
                identity = calls
            return FakeSystem(identity)

        return cache.get_or_create(key, build)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(get, range(16)))

    if scenario == "same-key":
        assert calls == 1
        assert len({id(result) for result in results}) == 1
    else:
        assert calls == 16
        assert len({id(result) for result in results}) == 16
