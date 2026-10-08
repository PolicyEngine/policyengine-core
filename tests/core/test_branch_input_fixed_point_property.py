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
