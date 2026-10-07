from __future__ import annotations

import pytest

from policyengine_core.caching import (
    CacheClosedError,
    InvalidCacheKeyError,
)
from policyengine_core.periods import period
from policyengine_core.simulations.simulation_result_cache import (
    ResultCacheKey,
    SimulationResultCache,
    SuppliedInputKey,
)


MONTHS = tuple(period(f"2024-{month:02d}") for month in range(1, 13))


@pytest.mark.parametrize(
    "operation,index",
    [
        ("put_get", 0),
        ("put_get", 1),
        ("put_get", 2),
        ("put_get", 3),
        ("replace", 4),
        ("replace", 5),
        ("replace", 6),
        ("replace", 7),
        ("discard", 8),
        ("discard", 9),
        ("discard_missing", 10),
        ("discard_missing", 11),
        ("clear", 0),
        ("clear", 4),
        ("missing_default", 8),
        ("missing_raises", 11),
    ],
    ids=lambda value: str(value),
)
def test_result_entry_contract(operation: str, index: int) -> None:
    cache: SimulationResultCache[object, str] = SimulationResultCache()
    key = ResultCacheKey(f"variable_{index}", MONTHS[index])

    if operation == "put_get":
        cache.put(key, f"value_{index}")
        assert cache.get(key) == f"value_{index}"
    elif operation == "replace":
        cache.put(key, "before")
        cache.put(key, "after")
        assert cache.get(key) == "after"
        assert cache.cache_info().writes == 2
    elif operation == "discard":
        cache.put(key, "value")
        assert cache.discard(key) is True
        assert cache.get(key, None) is None
    elif operation == "discard_missing":
        assert cache.discard(key) is False
        assert cache.cache_info().deletions == 0
    elif operation == "clear":
        cache.put(key, "value")
        cache.put(ResultCacheKey("other", MONTHS[(index + 1) % 12]), "other")
        cache.clear()
        assert len(cache) == 0
        assert cache.cache_info().deletions == 2
    elif operation == "missing_default":
        sentinel = object()
        assert cache.get(key, sentinel) is sentinel
        assert cache.cache_info().misses == 1
    else:
        with pytest.raises(KeyError, match=f"variable_{index}"):
            cache.get(key)


@pytest.mark.parametrize(
    "key",
    [
        None,
        ("salary", MONTHS[0]),
        ResultCacheKey("", MONTHS[0]),
        ResultCacheKey("salary", []),
        ResultCacheKey(1, MONTHS[0]),
        SuppliedInputKey("salary", "default", MONTHS[0]),
        "salary:2024-01",
        1,
    ],
    ids=[
        "none",
        "legacy-tuple",
        "empty-variable",
        "unhashable-period",
        "non-string-variable",
        "supplied-input-key",
        "string",
        "integer",
    ],
)
def test_result_key_validation_rejects_invalid_keys(key: object) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    with pytest.raises(InvalidCacheKeyError):
        cache.put(key, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "recorded_branch,visible_branches,expected",
    [
        ("default", ("default",), True),
        ("default", ("reform", "default"), True),
        ("default", ("nested", "reform", "default"), True),
        ("reform", ("reform", "default"), True),
        ("reform", ("nested", "reform", "default"), True),
        ("nested", ("nested", "reform", "default"), True),
        ("sibling", ("nested", "reform", "default"), False),
        ("baseline", ("reform", "default"), False),
        ("default", (), False),
        ("reform", ("default",), False),
        ("nested", ("reform", "default"), False),
        ("sibling", ("sibling",), True),
        ("branch-a", ("branch-a", "default"), True),
        ("branch-b", ("branch-a", "default"), False),
        ("", ("default",), False),
        ("default", ("default", "default"), True),
    ],
    ids=[
        "default-own",
        "default-visible-to-child",
        "default-visible-to-grandchild",
        "parent-own",
        "parent-visible-to-child",
        "child-own",
        "sibling-hidden",
        "baseline-hidden",
        "no-visible-branches",
        "child-hidden-from-default",
        "grandchild-hidden-from-parent",
        "sibling-own",
        "first-branch-own",
        "other-branch-hidden",
        "empty-record-not-created",
        "duplicate-visible-branches",
    ],
)
def test_supplied_input_periods_respect_branch_visibility(
    recorded_branch: str,
    visible_branches: tuple[str, ...],
    expected: bool,
) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    if recorded_branch:
        cache.record_supplied_input("salary", recorded_branch, MONTHS[0])

    assert cache.supplied_input_periods("salary", visible_branches) == (
        [MONTHS[0]] if expected else []
    )


@pytest.mark.parametrize(
    "delete_period,recorded_period,remove",
    [
        (None, MONTHS[0], True),
        (None, MONTHS[5], True),
        (period("2024"), MONTHS[0], True),
        (period("2024"), MONTHS[11], True),
        (period("2023"), MONTHS[0], False),
        (MONTHS[0], MONTHS[0], True),
        (MONTHS[0], MONTHS[1], False),
        (period("day:2024-01-01:15"), MONTHS[0], False),
        (period("day:2024-01-01:15"), period("2024-01-10"), True),
        (period("2024-01-10"), period("day:2024-01-01:15"), False),
        (period("eternity"), period("eternity"), True),
        (period("2024"), period("eternity"), False),
    ],
    ids=[
        "all-first-month",
        "all-middle-month",
        "year-contains-first-month",
        "year-contains-last-month",
        "other-year",
        "exact-month",
        "different-month",
        "partial-month-does-not-contain-month",
        "partial-month-contains-day",
        "day-does-not-contain-partial-month",
        "eternity-exact",
        "year-does-not-contain-eternity",
    ],
)
def test_discard_supplied_inputs_tracks_storage_deletion_scope(
    delete_period: object,
    recorded_period: object,
    remove: bool,
) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    key = cache.record_supplied_input("salary", "reform", recorded_period)

    removed = cache.discard_supplied_inputs(
        "salary",
        ("reform", "default"),
        delete_period,
    )

    assert (key not in cache.supplied_inputs) is remove
    assert removed == ({key} if remove else set())


@pytest.mark.parametrize(
    "operation,index",
    [
        ("mark", 0),
        ("mark", 1),
        ("duplicate", 2),
        ("duplicate", 3),
        ("take", 4),
        ("take", 5),
        ("discard", 6),
        ("discard", 7),
        ("discard-variable", 8),
        ("discard-variable", 9),
        ("clear", 10),
        ("clear", 11),
    ],
    ids=lambda value: str(value),
)
def test_invalidation_entries_are_owned_by_the_cache(
    operation: str, index: int
) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    key = ResultCacheKey(f"variable_{index}", MONTHS[index])

    if operation == "mark":
        cache.invalidate(key)
        assert cache.invalidated == {key}
    elif operation == "duplicate":
        cache.invalidate(key)
        cache.invalidate(key)
        assert cache.invalidated == {key}
    elif operation == "take":
        cache.invalidate(key)
        assert cache.take_invalidated() == {key}
        assert cache.invalidated == set()
    elif operation == "discard":
        cache.put(key, object())
        cache.invalidate(key)
        cache.discard_invalidated(key)
        assert key not in cache and key not in cache.invalidated
    elif operation == "discard-variable":
        cache.put(key, object())
        cache.put(ResultCacheKey(f"variable_{index}", MONTHS[0]), object())
        cache.discard_variable(f"variable_{index}")
        assert len(cache) == 0
    else:
        cache.put(key, object())
        cache.invalidate(key)
        cache.clear_calculated_results()
        assert len(cache) == 0 and not cache.invalidated


@pytest.mark.parametrize(
    "branches",
    [
        ("default",),
        ("reform",),
        ("nested",),
        ("default", "reform"),
        ("default", "reform", "nested"),
        ("a", "b", "a"),
        ("",),
        ("branch:with:separator",),
    ],
    ids=[
        "default",
        "reform",
        "nested",
        "nested-two",
        "nested-three",
        "repeated",
        "empty",
        "separator",
    ],
)
def test_supplied_input_context_stack_is_balanced(branches: tuple[str, ...]) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    entered: list[str] = []

    for branch in branches:
        if not branch or ":" in branch:
            with pytest.raises(InvalidCacheKeyError):
                with cache.supplied_input_context(branch):
                    pytest.fail("An invalid branch must never enter the context")
            assert cache.current_input_branch is None
            assert cache.input_revision == 0
            return
        with cache.supplied_input_context(branch):
            entered.append(cache.current_input_branch)
        assert cache.current_input_branch is None

    assert entered == list(branches)


@pytest.mark.parametrize(
    "preserve_results,preserve_invalidated,index",
    [
        (False, False, 0),
        (False, True, 1),
        (True, False, 2),
        (True, True, 3),
        (False, False, 4),
        (False, True, 5),
        (True, False, 6),
        (True, True, 7),
    ],
    ids=[
        "metadata-only-a",
        "metadata-invalidated-a",
        "results-a",
        "results-invalidated-a",
        "metadata-only-b",
        "metadata-invalidated-b",
        "results-b",
        "results-invalidated-b",
    ],
)
def test_clone_has_independent_indexes(
    preserve_results: bool,
    preserve_invalidated: bool,
    index: int,
) -> None:
    cache: SimulationResultCache[object, str] = SimulationResultCache()
    key = ResultCacheKey(f"variable_{index}", MONTHS[index])
    input_key = cache.record_supplied_input(
        f"variable_{index}", "default", MONTHS[index]
    )
    cache.put(key, "value")
    cache.invalidate(key)

    clone = cache.clone(
        preserve_results=preserve_results,
        preserve_invalidated=preserve_invalidated,
    )
    clone.supplied_inputs.clear()
    clone.discard(key)

    assert input_key in cache.supplied_inputs
    assert key in cache
    assert (key in clone) is False
    assert (key in clone.invalidated) is preserve_invalidated


@pytest.mark.parametrize(
    "operation,index",
    [
        ("close-get", 0),
        ("close-put", 1),
        ("close-discard", 2),
        ("close-length", 3),
        ("frozen-result-key", 4),
        ("frozen-input-key", 5),
        ("invalid-input-variable", 6),
        ("invalid-input-period", 7),
    ],
    ids=lambda value: str(value),
)
def test_prevented_result_cache_behavior(operation: str, index: int) -> None:
    cache: SimulationResultCache[object, object] = SimulationResultCache()
    key = ResultCacheKey(f"variable_{index}", MONTHS[index])

    if operation.startswith("close-"):
        cache.close()
        with pytest.raises(CacheClosedError):
            if operation == "close-get":
                cache.get(key)
            elif operation == "close-put":
                cache.put(key, object())
            elif operation == "close-discard":
                cache.discard(key)
            else:
                len(cache)
    elif operation == "frozen-result-key":
        with pytest.raises(AttributeError):
            key.variable_name = "other"  # type: ignore[misc]
    elif operation == "frozen-input-key":
        supplied = SuppliedInputKey("salary", "default", MONTHS[index])
        with pytest.raises(AttributeError):
            supplied.branch_name = "other"  # type: ignore[misc]
    elif operation == "invalid-input-variable":
        with pytest.raises(InvalidCacheKeyError):
            cache.record_supplied_input("", "default", MONTHS[index])
    else:
        with pytest.raises(InvalidCacheKeyError):
            cache.record_supplied_input("salary", "default", None)
