"""Differential coverage for repeated branch resets, including recreated branches."""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import example, given, settings, strategies as st

from tests.core.test_branch_input_fixed_point import (
    _assert_settled,
    _fallback_simulation,
    _input_first_result,
    _simulation,
)


@example(
    operations=[
        ("calculate", 2020, None),
        ("set", 2019, 123),
        ("calculate", 2020, None),
    ],
    cache_mode="memory",
)
@given(
    operations=st.lists(
        st.one_of(
            st.tuples(st.just("set"), st.integers(2018, 2022), st.integers(-500, 500)),
            st.tuples(st.just("calculate"), st.integers(2018, 2022), st.none()),
        ),
        min_size=1,
        max_size=25,
    ),
    cache_mode=st.sampled_from(["memory", "disk", "drop", "blacklist"]),
)
@settings(max_examples=100, deadline=None)
def test_fixed_point_fallback_never_shadows_later_input(operations, cache_mode):
    simulation, _ = _fallback_simulation(cache_mode)
    inputs = {}
    kept = {}
    for action, year, value in operations:
        if action == "set":
            simulation.set_input("carried", str(year), [value])
            inputs[year] = value
            kept[year] = value
            continue

        source = max((known for known in inputs if known <= year), default=None)
        # Root inputs keep previous input-derived calculations. Defaults
        # without a source must never acquire that persistent cache status.
        if year not in kept and source is not None:
            kept[year] = inputs[source]
        expected = kept.get(year, 0)
        assert simulation.calculate("carried", str(year)).tolist() == [expected]
        if source is None:
            assert simulation.get_array("carried", str(year)) is None


@given(
    values=st.lists(
        st.floats(min_value=-500, max_value=500, allow_nan=False, width=32),
        min_size=1,
        max_size=5,
    ),
    delete_derived=st.booleans(),
    recreate_branch=st.booleans(),
    cache_mode=st.sampled_from(["memory", "disk", "drop", "blacklist"]),
)
@settings(max_examples=40, deadline=None)
def test_branch_reset_fixed_point_matches_inputs_first(
    values, delete_derived, recreate_branch, cache_mode
):
    simulation, calls = _simulation(
        values,
        delete_derived=delete_derived,
        recreate_branch=recreate_branch,
        cache_mode=cache_mode,
    )
    _assert_settled(
        simulation,
        calls,
        _input_first_result(values),
        kept_in_holder=cache_mode in ("memory", "disk"),
    )


@given(
    values=st.lists(
        st.floats(min_value=-500, max_value=500, allow_nan=False, width=32),
        min_size=1,
        max_size=5,
    ),
    branch_depth=st.integers(min_value=1, max_value=3),
    cache_mode=st.sampled_from(["memory", "disk", "drop", "blacklist"]),
)
@settings(max_examples=40, deadline=None)
def test_reused_and_recreated_branch_paths_have_equal_results_and_work(
    values, branch_depth, cache_mode
):
    # Persisting a branch path and recreating it must both match inputs-first
    # values, settle in exactly two attempts, and reuse the settled result
    # while respecting holder storage restrictions.
    expected = _input_first_result(values)
    for recreate_branch in (False, True):
        simulation, calls = _simulation(
            values,
            branch_depth=branch_depth,
            recreate_branch=recreate_branch,
            cache_mode=cache_mode,
        )
        _assert_settled(
            simulation,
            calls,
            expected,
            kept_in_holder=cache_mode in ("memory", "disk"),
        )
