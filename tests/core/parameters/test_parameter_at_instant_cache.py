from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass

import pytest

from policyengine_core.caching import (
    CacheClosedError,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.parameters import LazyParameterNodeAtInstant, ParameterNode
from policyengine_core.parameters.parameter_at_instant_cache import (
    ParameterAtInstantCache,
)
from policyengine_core.periods import instant
from policyengine_core.tracers import FullTracer, TracingParameterNodeAtInstant


@dataclass
class FakeView:
    value: int


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("insert", id="insert"),
        pytest.param("lookup", id="lookup"),
        pytest.param("replace", id="replace"),
        pytest.param("delete", id="delete"),
        pytest.param("discard-missing", id="discard-missing"),
        pytest.param("clear-empty", id="clear-empty"),
        pytest.param("clear-populated", id="clear-populated"),
        pytest.param("contains-present", id="contains-present"),
        pytest.param("contains-absent", id="contains-absent"),
        pytest.param("iteration", id="iteration"),
        pytest.param("length", id="length"),
        pytest.param("getitem", id="getitem"),
        pytest.param("setitem", id="setitem"),
        pytest.param("delitem", id="delitem"),
        pytest.param("missing-default", id="missing-default"),
        pytest.param("missing-error", id="missing-error"),
    ],
)
def test_parameter_cache_mapping_contract(scenario: str) -> None:
    cache = ParameterAtInstantCache[str, FakeView]()
    first = FakeView(1)
    second = FakeView(2)
    if scenario not in {
        "clear-empty",
        "contains-absent",
        "missing-default",
        "missing-error",
    }:
        cache.put("2026-01-01", first)

    if scenario == "insert":
        assert cache.cache_info().writes == 1
    elif scenario == "lookup":
        assert cache.get("2026-01-01") is first
    elif scenario == "replace":
        cache.put("2026-01-01", second)
        assert cache.get("2026-01-01") is second
    elif scenario == "delete":
        assert cache.discard("2026-01-01") is True
    elif scenario == "discard-missing":
        assert cache.discard("missing") is False
    elif scenario.startswith("clear"):
        cache.clear()
        assert len(cache) == 0
    elif scenario == "contains-present":
        assert "2026-01-01" in cache
    elif scenario == "contains-absent":
        assert "missing" not in cache
    elif scenario == "iteration":
        assert list(cache) == ["2026-01-01"]
    elif scenario == "length":
        assert len(cache) == 1
    elif scenario == "getitem":
        assert cache["2026-01-01"] is first
    elif scenario == "setitem":
        cache["2026-02-01"] = second
        assert cache["2026-02-01"] is second
    elif scenario == "delitem":
        del cache["2026-01-01"]
        assert len(cache) == 0
    elif scenario == "missing-default":
        assert cache.get("missing", second) is second
    else:
        with pytest.raises(KeyError):
            cache.get("missing")


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("2026-01-01", id="string"),
        pytest.param(instant("2026-01-01"), id="instant"),
        pytest.param((2026, 1, 1), id="tuple"),
        pytest.param(2026, id="integer"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(b"2026-01-01", id="bytes"),
        pytest.param(frozenset({"2026"}), id="frozenset"),
    ],
)
def test_parameter_cache_accepts_hashable_instant_keys(key) -> None:
    cache = ParameterAtInstantCache[object, FakeView]()
    view = FakeView(1)

    cache.put(key, view)

    assert cache.get(key) is view


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("enabled-miss", id="enabled-miss"),
        pytest.param("enabled-hit", id="enabled-hit"),
        pytest.param("disabled-first", id="disabled-first"),
        pytest.param("disabled-repeated", id="disabled-repeated"),
    ],
)
def test_parameter_cache_reference_mode(scenario: str) -> None:
    cache = ParameterAtInstantCache[str, FakeView](
        enabled=not scenario.startswith("disabled")
    )
    calls = 0

    def build() -> FakeView:
        nonlocal calls
        calls += 1
        return FakeView(calls)

    first = cache.get_or_create("2026-01-01", build)
    second = cache.get_or_create("2026-01-01", build)

    if scenario.startswith("disabled"):
        assert (first.value, second.value, calls, len(cache)) == (1, 2, 2, 0)
    else:
        assert first is second
        assert calls == 1


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        pytest.param("key", [], id="list-key"),
        pytest.param("key", {}, id="mapping-key"),
        pytest.param("key", set(), id="set-key"),
        pytest.param("key", None, id="none-key"),
        pytest.param("value", None, id="none-value"),
        pytest.param("factory", None, id="none-factory-value"),
        pytest.param("closed", "get", id="get-after-close"),
        pytest.param("closed", "put", id="put-after-close"),
    ],
)
def test_parameter_cache_prevents_invalid_operations(kind: str, value) -> None:
    cache = ParameterAtInstantCache[object, FakeView]()
    if kind == "key":
        with pytest.raises(InvalidCacheKeyError):
            cache.put(value, FakeView(1))
    elif kind == "value":
        with pytest.raises(InvalidCacheValueError):
            cache.put("2026-01-01", value)
    elif kind == "factory":
        with pytest.raises(InvalidCacheValueError):
            cache.get_or_create("2026-01-01", lambda: value)
    else:
        cache.close()
        with pytest.raises(CacheClosedError):
            if value == "get":
                cache.get("2026-01-01")
            else:
                cache.put("2026-01-01", FakeView(1))


def make_node() -> ParameterNode:
    return ParameterNode(
        "root",
        data={
            "rate": {"values": {"2020-01-01": 0.2, "2026-01-01": 0.3}},
            "amount": {"values": {"2020-01-01": 100, "2026-01-01": 200}},
        },
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("untraced-first", id="untraced-first"),
        pytest.param("traced-first", id="traced-first"),
        pytest.param("traced-after-untraced", id="traced-after-untraced"),
        pytest.param("untraced-after-traced", id="untraced-after-traced"),
        pytest.param("two-tracers", id="two-tracers"),
        pytest.param("two-branches", id="two-branches"),
        pytest.param("two-instants", id="two-instants"),
        pytest.param("repeated-traced", id="repeated-traced"),
        pytest.param("cached-value-plain", id="cached-value-plain"),
        pytest.param("wrapper-current-tracer", id="wrapper-current-tracer"),
        pytest.param("wrapper-current-branch", id="wrapper-current-branch"),
        pytest.param("values-equal", id="values-equal"),
        pytest.param("children-plain", id="children-plain"),
        pytest.param("cache-size", id="cache-size"),
        pytest.param("trace-off", id="trace-off"),
        pytest.param("trace-on", id="trace-on"),
    ],
)
def test_parameter_node_cache_is_tracer_neutral(scenario: str) -> None:
    node = make_node()
    first_tracer = FullTracer()
    second_tracer = FullTracer()

    plain = node("2026-01-01")
    node.trace = True
    node.tracer = first_tracer
    node.branch_name = "first"
    traced = node("2026-01-01")
    if scenario in {"two-tracers", "wrapper-current-tracer"}:
        node.tracer = second_tracer
    if scenario in {"two-branches", "wrapper-current-branch"}:
        node.branch_name = "second"
    traced_again = node("2026-01-01")
    node.trace = False
    untraced_again = node("2026-01-01")

    assert type(plain) is LazyParameterNodeAtInstant
    assert type(traced) is TracingParameterNodeAtInstant
    assert type(traced_again) is TracingParameterNodeAtInstant
    assert type(untraced_again) is LazyParameterNodeAtInstant
    assert traced.parameter_node_at_instant is plain
    assert traced_again.parameter_node_at_instant is plain
    assert all(
        type(value) is LazyParameterNodeAtInstant
        for value in node.parameter_cache.values()
    )
    if scenario == "wrapper-current-tracer":
        assert traced_again.tracer is second_tracer
    if scenario == "wrapper-current-branch":
        assert traced_again.branch_name == "second"
    if scenario == "values-equal":
        assert traced.rate == untraced_again.rate == 0.3


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("shallow-independent-index", id="shallow-independent-index"),
        pytest.param("shallow-shared-value", id="shallow-shared-value"),
        pytest.param("deep-independent-index", id="deep-independent-index"),
        pytest.param("deep-independent-value", id="deep-independent-value"),
        pytest.param("pickle-round-trip", id="pickle-round-trip"),
        pytest.param("pickle-independent-index", id="pickle-independent-index"),
        pytest.param("copy-open", id="copy-open"),
        pytest.param("deepcopy-open", id="deepcopy-open"),
        pytest.param("pickle-open", id="pickle-open"),
        pytest.param("copy-metrics-reset", id="copy-metrics-reset"),
        pytest.param("deepcopy-metrics-reset", id="deepcopy-metrics-reset"),
        pytest.param("pickle-metrics-reset", id="pickle-metrics-reset"),
    ],
)
def test_parameter_cache_copy_and_pickle_contract(scenario: str) -> None:
    cache = ParameterAtInstantCache[str, FakeView]()
    original = FakeView(1)
    cache.put("2026-01-01", original)
    if scenario.startswith("shallow") or scenario.startswith("copy"):
        duplicate = copy.copy(cache)
    elif scenario.startswith("deep"):
        duplicate = copy.deepcopy(cache)
    else:
        duplicate = pickle.loads(pickle.dumps(cache))

    assert duplicate.get("2026-01-01").value == 1
    assert duplicate.cache_info().closed is False
    if "independent-index" in scenario:
        duplicate.discard("2026-01-01")
        assert "2026-01-01" in cache
    if scenario == "shallow-shared-value":
        assert duplicate.get("2026-01-01") is original
    if scenario == "deep-independent-value":
        assert duplicate.get("2026-01-01") is not original
    if "metrics-reset" in scenario:
        info = duplicate.cache_info()
        assert (info.hits, info.misses, info.builds) == (1, 0, 0)


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("empty-mapping-assignment", id="empty-mapping-assignment"),
        pytest.param("populated-mapping-assignment", id="populated-mapping-assignment"),
        pytest.param("cache-assignment", id="cache-assignment"),
        pytest.param("tracing-wrapper-assignment", id="tracing-wrapper-assignment"),
        pytest.param("clear-local", id="clear-local"),
        pytest.param("clear-recursive", id="clear-recursive"),
        pytest.param("clone-empty", id="clone-empty"),
        pytest.param("clone-independent", id="clone-independent"),
    ],
)
def test_parameter_node_compatibility_and_clear_operations(scenario: str) -> None:
    node = make_node()
    dated = node("2026-01-01")
    if scenario == "empty-mapping-assignment":
        with pytest.raises(AttributeError):
            node._at_instant_cache = {}
        assert node("2026-01-01") is dated
    elif scenario == "populated-mapping-assignment":
        with pytest.raises(AttributeError):
            node._at_instant_cache = {instant("2026-01-01"): dated}
        assert node("2026-01-01") is dated
    elif scenario == "cache-assignment":
        cache = ParameterAtInstantCache()
        with pytest.raises(AttributeError):
            node.parameter_cache = cache
        assert node.parameter_cache is not cache
    elif scenario == "tracing-wrapper-assignment":
        wrapped = TracingParameterNodeAtInstant(dated, FullTracer(), "default")
        with pytest.raises(AttributeError):
            node._at_instant_cache = {instant("2026-01-01"): wrapped}
        assert node("2026-01-01") is dated
    elif scenario.startswith("clear"):
        node.clear_at_instant_caches(recursive=scenario == "clear-recursive")
        assert len(node.parameter_cache) == 0
    else:
        cloned = node.clone()
        assert len(cloned.parameter_cache) == 0
        if scenario == "clone-independent":
            assert cloned.parameter_cache is not node.parameter_cache


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("clear-system", id="clear-system"),
        pytest.param("clear-tree", id="clear-tree"),
        pytest.param("share-root", id="share-root"),
        pytest.param("share-system-cache", id="share-system-cache"),
        pytest.param("share-node-cache", id="share-node-cache"),
        pytest.param("clone-independent", id="clone-independent"),
        pytest.param("mapping-compatibility", id="mapping-compatibility"),
        pytest.param("cached-values-plain", id="cached-values-plain"),
    ],
)
def test_tax_benefit_system_parameter_cache_operations(scenario: str) -> None:
    system = CountryTaxBenefitSystem()
    dated = system.get_parameters_at_instant("2026-01-01")
    if scenario.startswith("clear"):
        system.clear_parameter_caches()
        assert len(system.parameter_cache) == 0
        assert len(system.parameters.parameter_cache) == 0
    elif scenario.startswith("share"):
        other = CountryTaxBenefitSystem()
        other.share_parameters_from(system)
        assert other.parameters is system.parameters
        assert other.parameter_cache is system.parameter_cache
        if scenario == "share-node-cache":
            assert other.parameters.parameter_cache is system.parameters.parameter_cache
    elif scenario == "clone-independent":
        cloned = system.clone()
        assert cloned.parameter_cache is not system.parameter_cache
    elif scenario == "mapping-compatibility":
        with pytest.raises(AttributeError):
            system._parameters_at_instant_cache = {instant("2026-01-01"): dated}
        assert system.get_parameters_at_instant("2026-01-01") is dated
    else:
        assert all(
            type(value) is LazyParameterNodeAtInstant
            for value in system.parameter_cache.values()
        )
