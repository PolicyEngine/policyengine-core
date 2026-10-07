"""Generic cache guarantees available before transitional APIs are removed."""

from collections.abc import Mapping

import numpy as np
import pytest

from policyengine_core.caching import (
    CacheClosedError,
    InvalidCacheKeyError,
    InvalidCacheValueError,
)
from policyengine_core.data_storage.immutable_array_cache import (
    CachedArrayEntry,
    ImmutableArrayCache,
)
from policyengine_core.periods import period
from policyengine_core.simulations.simulation_result_cache import (
    ResultCacheKey,
    SimulationResultCache,
    SuppliedInputKey,
)


MONTH = period("2025-01")
RESULT_KEY = ResultCacheKey("income", MONTH)
INPUT_KEY = SuppliedInputKey("income", "default", MONTH)


@pytest.fixture(params=["arrays", "results"], ids=["immutable-arrays", "results"])
def populated_cache(request):
    if request.param == "arrays":
        cache, key, value = (
            ImmutableArrayCache(),
            "stored",
            CachedArrayEntry(np.array([4])),
        )
    else:
        cache, key, value = SimulationResultCache(), RESULT_KEY, "calculated"
    cache.put(key, value)
    return cache, key, value


def test_bulk_entries_reject_closed_cache_without_mutation(populated_cache):
    cache, key, value = populated_cache
    cache.close()
    before = cache.cache_info()
    with pytest.raises(CacheClosedError):
        cache.replace_entries({key: value})
    assert not cache.entries
    assert cache.cache_info() == before


def _key(cache, name):
    return (
        ResultCacheKey(name, MONTH)
        if isinstance(cache, SimulationResultCache)
        else name
    )


def test_bulk_entries_count_overlap_removal_and_empty_replacement(populated_cache):
    cache, retained, value = populated_cache
    cache.put(_key(cache, "removed"), value)
    before = cache.cache_info()
    cache.replace_entries(
        {
            retained: value,
            _key(cache, "first_new"): value,
            _key(cache, "second_new"): value,
        }
    )
    after = cache.cache_info()
    assert after.writes == before.writes + 3
    assert after.deletions == before.deletions + 1
    assert after.entries == 3
    assert after.peak_entries == max(before.peak_entries, 3)
    assert (after.hits, after.misses, after.builds, after.evictions) == (
        before.hits,
        before.misses,
        before.builds,
        before.evictions,
    )
    cache.replace_entries({})
    empty = cache.cache_info()
    assert empty.entries == 0
    assert empty.writes == after.writes
    assert empty.deletions == after.deletions + 3
    assert empty.peak_entries == after.peak_entries


class _UncopyableValue:
    def __deepcopy__(self, memo):
        raise RuntimeError("cannot construct immutable value")


def test_failed_bulk_validation_or_construction_preserves_metrics_and_entries(
    populated_cache,
):
    cache, key, value = populated_cache
    before = cache.cache_info()
    if isinstance(cache, SimulationResultCache):
        bad_key, bad_value = ResultCacheKey("", MONTH), "bad"
        expected_error = InvalidCacheKeyError
    else:
        bad_key, bad_value = "bad", _UncopyableValue()
        # Preparatory conversion can fail during construction; a later strict
        # typed API can reject the same unsupported value before construction.
        expected_error = (RuntimeError, InvalidCacheValueError)
    with pytest.raises(expected_error):
        cache.replace_entries({key: value, bad_key: bad_value})
    assert cache.cache_info() == before
    assert tuple(cache.entries) == (key,)
    assert cache.entries[key] is value


class _LockObserver:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, *exception):
        self.depth -= 1


def test_bulk_entries_validate_inside_owner_lock(populated_cache, monkeypatch):
    cache, key, value = populated_cache
    lock = _LockObserver()
    monkeypatch.setattr(cache, "_cache_lock", lock)

    class CheckedMapping(dict):
        def items(self):
            assert lock.depth > 0
            return super().items()

    cache.replace_entries(CheckedMapping({key: value}))
    assert lock.entries > 0
    assert lock.depth == 0
    assert cache.entries[key] is value


@pytest.mark.parametrize(
    "method,key",
    [("replace_invalidated", RESULT_KEY), ("replace_supplied_inputs", INPUT_KEY)],
    ids=["pending-invalidations", "supplied-inputs"],
)
def test_bulk_metadata_validates_under_lock_without_result_counters(
    method, key, monkeypatch
):
    cache = SimulationResultCache()
    cache.put(RESULT_KEY, "original")
    before = cache.cache_info()
    lock = _LockObserver()
    monkeypatch.setattr(cache, "_cache_lock", lock)

    class CheckedSet(set):
        def __iter__(self):
            assert lock.depth > 0
            return super().__iter__()

    getattr(cache, method)(CheckedSet({key}))
    assert lock.entries > 0
    assert lock.depth == 0
    assert cache.cache_info() == before


def _metadata_state(cache):
    return (
        dict(cache.entries),
        frozenset(cache.invalidated),
        cache.supplied_input_keys(),
        tuple(cache.input_contexts),
        cache.input_revision,
        frozenset(cache._result_variables),
        cache.cache_info(),
    )


def _mutate(cache, operation):
    other_result = ResultCacheKey("other", MONTH)
    other_input = SuppliedInputKey("other", "default", MONTH)
    return {
        "replace_invalidated": lambda: cache.replace_invalidated({other_result}),
        "replace_supplied_inputs": lambda: cache.replace_supplied_inputs({other_input}),
        "record_supplied_input": lambda: cache.record_supplied_input(*other_input),
        "forget_supplied_input": lambda: cache.forget_supplied_input(*INPUT_KEY),
        "retain_input_variables": lambda: cache.retain_input_variables(set()),
        "discard_supplied_inputs": lambda: cache.discard_supplied_inputs(
            "income", ("default",), MONTH
        ),
        "record_result_storage": lambda: cache.record_result_storage("other"),
        "take_result_variables": cache.take_result_variables,
        "invalidate": lambda: cache.invalidate(other_result),
        "take_invalidated": cache.take_invalidated,
        "discard_invalidated": lambda: cache.discard_invalidated(RESULT_KEY),
        "discard_variable": lambda: cache.discard_variable("income"),
        "discard_result_entries": lambda: cache.discard_result_entries("income"),
        "clear_calculated_results": cache.clear_calculated_results,
    }[operation]()


@pytest.mark.parametrize(
    "operation",
    [
        "replace_invalidated",
        "replace_supplied_inputs",
        "record_supplied_input",
        "forget_supplied_input",
        "retain_input_variables",
        "discard_supplied_inputs",
        "record_result_storage",
        "take_result_variables",
        "invalidate",
        "take_invalidated",
        "discard_invalidated",
        "discard_variable",
        "discard_result_entries",
        "clear_calculated_results",
    ],
    ids=[
        "replace-pending",
        "replace-provenance",
        "record-provenance",
        "forget-provenance",
        "prune-provenance",
        "delete-provenance",
        "record-result-holder",
        "consume-result-holders",
        "record-invalidation",
        "consume-invalidations",
        "remove-one-invalidation",
        "remove-variable",
        "remove-fast-results",
        "clear-results-and-pending",
    ],
)
def test_closed_metadata_mutation_preserves_every_index_and_revision(operation):
    cache = SimulationResultCache()
    cache.put(RESULT_KEY, "original")
    cache.invalidate(RESULT_KEY)
    cache.record_supplied_input(*INPUT_KEY)
    cache.record_result_storage("income")
    cache.close()
    before = _metadata_state(cache)
    with pytest.raises(CacheClosedError):
        _mutate(cache, operation)
    assert _metadata_state(cache) == before


def test_closed_cache_rejects_context_entry_without_pushing_branch():
    cache = SimulationResultCache()
    cache.close()
    with pytest.raises(CacheClosedError):
        with cache.supplied_input_context("branch"):
            pytest.fail("closed cache entered input context")
    assert tuple(cache.input_contexts) == ()


def test_nested_contexts_unwind_even_when_cache_closes_inside_context(monkeypatch):
    cache = SimulationResultCache()
    lock = _LockObserver()
    monkeypatch.setattr(cache, "_cache_lock", lock)
    with cache.supplied_input_context("parent"):
        assert lock.depth > 0
        with cache.supplied_input_context("child"):
            assert lock.depth > 1
            cache.close()
        assert tuple(cache.input_contexts) == ("parent",)
    assert tuple(cache.input_contexts) == ()
    assert lock.depth == 0
    assert cache.cache_info().closed


@pytest.mark.parametrize(
    "query",
    [
        "entries",
        "invalidated",
        "supplied_inputs",
        "supplied_input_keys",
        "has_supplied_input",
        "input_contexts",
        "current_input_branch",
        "input_revision",
        "supplied_input_periods",
    ],
    ids=[
        "result-index",
        "pending-index",
        "provenance-index",
        "provenance-query",
        "exact-input-query",
        "context-index",
        "active-context",
        "input-revision",
        "visible-periods",
    ],
)
def test_result_observation_holds_lock_during_query_not_caller_iteration(
    query, monkeypatch
):
    cache = SimulationResultCache()
    cache.put(RESULT_KEY, "original")
    cache.invalidate(RESULT_KEY)
    cache.record_supplied_input(*INPUT_KEY)
    lock = _LockObserver()
    monkeypatch.setattr(cache, "_cache_lock", lock)
    for close_first in (False, True):
        if close_first:
            cache.close()
        previous_entries = lock.entries
        if query == "supplied_input_keys":
            observation = cache.supplied_input_keys()
        elif query == "has_supplied_input":
            observation = cache.has_supplied_input(*INPUT_KEY)
        elif query == "supplied_input_periods":
            observation = cache.supplied_input_periods("income", ("default",))
        else:
            observation = getattr(cache, query)
        assert lock.entries > previous_entries
        assert lock.depth == 0
        if isinstance(observation, (Mapping, tuple, frozenset, list, set)):
            tuple(observation)
            assert lock.depth == 0


def test_array_entries_observation_holds_lock_only_during_lookup(monkeypatch):
    cache = ImmutableArrayCache()
    cache.put_array("stored", np.array([4]))
    lock = _LockObserver()
    monkeypatch.setattr(cache, "_cache_lock", lock)
    observation = cache.entries
    assert lock.entries > 0
    assert lock.depth == 0
    np.testing.assert_array_equal(observation["stored"].read(), [4])
    cache.close()
    previous_entries = lock.entries
    assert not cache.entries
    assert lock.entries > previous_entries
    assert lock.depth == 0
