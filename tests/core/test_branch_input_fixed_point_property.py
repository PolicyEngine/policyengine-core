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
@example(
    operations=[
        ("set", 2019, 1),
        ("calculate", 2018, None),
        ("calculate", "2020-01", None),
        ("set", 2019, 0),
        ("calculate", 2020, None),
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
        # converted month or its source year. Derive each expected value from
        # explicit inputs and previously modeled values, never cache contents.
        annual = kept.get(period.this_year, inputs[source] if source is not None else 0)
        converted = annual / 12 if period.unit == periods.MONTH else annual
        expected = kept.get(period, converted)
        assert simulation.calculate("carried", period).tolist() == pytest.approx(
            [expected]
        )

        # A reused worker can leave the source year fresh enough to cache;
        # drop/blacklist modes can omit the converted month. Observe only
        # whether caches retain the independently modeled values, validating
        # every retained value before remembering its expected counterpart.
        candidates = {period.this_year: annual, period: expected}
        for candidate, modeled in candidates.items():
            stored = simulation.get_array("carried", candidate)
            fast = simulation._fast_cache.get(("carried", candidate))
            for cached in (stored, fast):
                if cached is not None:
                    assert cached.tolist() == pytest.approx([modeled])
            if stored is not None or fast is not None:
                kept[candidate] = modeled
            else:
                kept.pop(candidate, None)
        if source is None:
            assert simulation.get_array("carried", period) is None
            assert simulation.get_array("carried", period.this_year) is None
            assert ("carried", period) not in simulation._fast_cache
            assert ("carried", period.this_year) not in simulation._fast_cache


@example(
    definition_period=periods.YEAR,
    use_option=False,
    month=2,
    value=123,
    supplied_during_calculation=True,
    cache_mode="memory",
)
@example(
    definition_period=periods.MONTH,
    use_option=False,
    month=6,
    value=120,
    supplied_during_calculation=True,
    cache_mode="memory",
)
@example(
    definition_period=periods.MONTH,
    use_option=True,
    month=6,
    value=120,
    supplied_during_calculation=True,
    cache_mode="memory",
)
@given(
    definition_period=st.sampled_from([periods.YEAR, periods.MONTH]),
    use_option=st.booleans(),
    month=st.integers(2, 12),
    value=st.integers(-500, 500),
    supplied_during_calculation=st.booleans(),
    cache_mode=st.sampled_from(["memory", "disk", "drop", "blacklist"]),
)
@settings(max_examples=100, deadline=None)
def test_converted_fallback_matches_input_supplied_during_calculation(
    definition_period,
    use_option,
    month,
    value,
    supplied_during_calculation,
    cache_mode,
):
    simulation, _ = _fallback_simulation(cache_mode, definition_period)
    variable = simulation.tax_benefit_system.get_variable("carried")
    original = variable.get_formula("2020")
    requested = f"2020-{month:02d}" if definition_period == periods.YEAR else "2020"
    supplied = False

    def carried(person, period):
        nonlocal supplied
        original(person, period)
        if (
            supplied_during_calculation
            and not supplied
            and (definition_period == periods.YEAR or period.start.month == month)
        ):
            person.simulation.set_input("carried", requested, [value])
            # Retrying the sum must still recognize this first-attempt input
            # without relying on the formula to write it again.
            supplied = True
        # Returning None still selects fallback; a newly supplied input for
        # the requested period takes precedence over the converted result.

    # Preserve the variable shared with any holders already constructed for
    # disk/drop storage, including the input handler those holders consult.
    variable.formulas["2020-01-01"] = carried
    variable.set_input = None
    option = (
        "calculate_divide" if definition_period == periods.YEAR else "calculate_add"
    )
    method = option if use_option else "calculate"

    # The model depends only on whether the formula supplies the requested
    # input. In particular, it never divides that input or sums carried terms.
    expected = value if supplied_during_calculation else 0
    assert getattr(simulation, method)("carried", requested).tolist() == pytest.approx(
        [expected]
    )
    if supplied_during_calculation:
        assert simulation.get_array("carried", requested).tolist() == [value]
    else:
        assert (
            simulation.get_holder("carried").get_known_periods(simulation.branch_name)
            == []
        )
        assert not any(name == "carried" for name, _ in simulation._fast_cache)
    assert not simulation._open_frames


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
