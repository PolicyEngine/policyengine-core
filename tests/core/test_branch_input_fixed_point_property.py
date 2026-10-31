"""Differential coverage for repeated branch resets, including recreated branches."""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from tests.core.test_branch_input_fixed_point import (
    _assert_settled,
    _input_first_result,
    _simulation,
)


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
