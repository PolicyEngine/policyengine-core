"""Differential coverage for repeated branch resets, including recreated branches."""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import example, given, settings, strategies as st

from policyengine_core import periods

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
@example(
    operations=[
        ("calculate", "2020-01", None),
        ("set", 2019, 123),
        ("calculate", "2020-01", None),
    ],
    cache_mode="memory",
)
@example(
    operations=[
        ("set", 2018, 0),
        ("calculate", "2018-01", None),
        ("set", 2018, 1),
        ("calculate", "2018-01", None),
    ],
    cache_mode="memory",
)
@given(
    operations=st.lists(
        st.one_of(
            st.tuples(st.just("set"), st.integers(2018, 2022), st.integers(-500, 500)),
            st.tuples(
                st.just("calculate"),
                st.one_of(
                    st.integers(2018, 2022),
                    st.tuples(st.integers(2018, 2022), st.integers(1, 12)).map(
                        lambda value: f"{value[0]}-{value[1]:02d}"
                    ),
                ),
                st.none(),
            ),
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
    for action, requested, value in operations:
        period = periods.period(requested)
        year = period.start.year
        if action == "set":
            simulation.set_input("carried", period, [value])
            inputs[year] = value
            # Writing an annual input removes overlapping calculated months
            # of this variable, even on the root (independent of branch
            # dependency invalidation).
            kept = {
                known: result
                for known, result in kept.items()
                if known.start.year != year
            }
            kept[period] = value
            continue

        source = max((known for known in inputs if known <= year), default=None)
        # Root inputs keep previous input-derived calculations, including a
        # converted month. A monthly request does not itself retain the year
        # while the branch reset makes its underlying calculation stale.
        if period not in kept and source is not None:
            annual = kept.get(period.this_year, inputs[source])
            kept[period] = annual / 12 if period.unit == periods.MONTH else annual
        expected = kept.get(period, 0)
        assert simulation.calculate("carried", period).tolist() == pytest.approx(
            [expected]
        )
        if source is None:
            assert simulation.get_array("carried", period) is None
            assert simulation.get_array("carried", period.this_year) is None


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
