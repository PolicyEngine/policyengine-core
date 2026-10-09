"""Properties of the fast cache when periods are deleted or set.

Invariants (each tested for random inputs):

1. The fast cache never changes what ``calculate`` returns. Random sequences
   of ``set_input``, ``calculate`` and ``delete_arrays`` run on two
   simulations, one of which empties its fast cache before every
   ``calculate`` (so every value comes from storage or a formula). Every
   result, and every stored value at the end, must agree.
2. ``delete_arrays`` drops from the fast cache exactly the periods it drops
   from holder storage: afterwards the variable's fast-cache periods equal its
   stored periods, whatever was in both before, and other variables' entries
   are untouched.

``test_fast_cache_contained_periods.py`` pins the same behaviour with
examples.
"""

from __future__ import annotations

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from tests.fixtures.fast_cache_contained_periods import build_simulation

MONTHS = ["2012-01", "2012-02", "2012-12", "2013-01", "2013-02"]
YEARS = ["2012", "2013"]
ANY_PERIOD = MONTHS + YEARS

# (variable, periods ``set_input`` may use)
SET_INPUTS = [
    ("monthly_input", MONTHS),
    ("monthly_input_split", MONTHS + YEARS),
    ("yearly_input", YEARS),
    ("eternal_input", ANY_PERIOD),
]
# (variable, periods ``calculate`` may use); a monthly variable read for a
# year is summed, a yearly one read for a month is divided.
READS = [
    ("monthly_input", ANY_PERIOD),
    ("monthly_input_split", ANY_PERIOD),
    ("monthly_formula", ANY_PERIOD),
    ("yearly_input", ANY_PERIOD),
    ("eternal_input", ANY_PERIOD),
    ("eternal_formula", ANY_PERIOD),
]
DELETE_PERIODS = [None, periods.ETERNITY] + ANY_PERIOD

_value = st.sampled_from([0.0, 1.5, 100.0, 1200.0])


def _operation_over(variables):
    """``set_input``, ``calculate`` or ``delete_arrays`` on one of ``variables``."""
    set_inputs = [(name, ps) for name, ps in SET_INPUTS if name in variables]
    reads = [(name, ps) for name, ps in READS if name in variables]
    options = [
        st.sampled_from(reads).flatmap(
            lambda pair: st.tuples(
                st.just("calculate"), st.just(pair[0]), st.sampled_from(pair[1])
            )
        ),
        st.tuples(
            st.just("delete_arrays"),
            st.sampled_from([name for name, _ in reads]),
            st.sampled_from(DELETE_PERIODS),
        ),
    ]
    if set_inputs:
        options.append(
            st.sampled_from(set_inputs).flatmap(
                lambda pair: st.tuples(
                    st.just("set_input"),
                    st.just(pair[0]),
                    st.sampled_from(pair[1]),
                    _value,
                )
            )
        )
    return st.one_of(*options)


# Sequences over every variable, or over one family (an input and what is
# calculated from it), where the same values are hit again more often.
FAMILIES = [
    ("monthly_input", "monthly_formula"),
    ("monthly_input_split",),
    ("yearly_input",),
    ("eternal_input", "eternal_formula"),
]
_operations = st.one_of(
    st.lists(_operation_over([name for name, _ in READS]), max_size=25),
    st.sampled_from(FAMILIES).flatmap(
        lambda family: st.lists(_operation_over(family), max_size=15)
    ),
)


def _result(call):
    """What a call returns, or the type of error it raises."""
    try:
        value = call()
    except Exception as error:  # Compared across simulations, not hidden.
        return ("raised", type(error).__name__)
    if value is None:
        return ("none",)
    return ("value", np.asarray(value).tolist())


def _apply(simulation, operation, empty_fast_cache_first=False):
    kind, variable, period, *rest = operation
    if kind == "set_input":
        return _result(lambda: simulation.set_input(variable, period, [rest[0]]))
    if kind == "calculate":
        if empty_fast_cache_first:
            simulation._fast_cache.clear()
        return _result(lambda: simulation.calculate(variable, period))
    return _result(lambda: simulation.delete_arrays(variable, period))


@hypothesis.settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(operations=_operations)
def test_fast_cache_never_changes_what_calculate_returns(operations):
    cached = build_simulation()
    uncached = build_simulation()

    for step, operation in enumerate(operations):
        expected = _apply(uncached, operation, empty_fast_cache_first=True)
        assert _apply(cached, operation) == expected, (step, operation)

    for variable, read_periods in READS:
        for period in read_periods:
            expected = _result(lambda: uncached.get_array(variable, period))
            assert _result(lambda: cached.get_array(variable, period)) == expected


_cached_period = st.sampled_from(
    MONTHS + YEARS + ["2012-02-15", "month:2012-11:3", "year:2012:2", "2011"]
).map(periods.period)


@hypothesis.settings(max_examples=300, deadline=None)
@hypothesis.given(
    cached=st.sets(_cached_period, max_size=8),
    other_variable_cached=st.sets(_cached_period, max_size=3),
    deleted=st.sampled_from(DELETE_PERIODS),
)
def test_delete_arrays_drops_from_the_fast_cache_what_it_drops_from_storage(
    cached, other_variable_cached, deleted
):
    simulation = build_simulation(carry_over=False)
    holder = simulation.get_holder("monthly_formula")
    for period in cached:
        value = np.asarray([1.0])
        holder._memory_storage.put(value, period, "default")
        simulation._fast_cache[("monthly_formula", period)] = value
    for period in other_variable_cached:
        simulation._fast_cache[("monthly_input", period)] = np.asarray([2.0])

    simulation.delete_arrays("monthly_formula", deleted)

    in_fast_cache = {
        period for name, period in simulation._fast_cache if name == "monthly_formula"
    }
    assert in_fast_cache == set(holder.get_known_periods())
    assert {
        period for name, period in simulation._fast_cache if name == "monthly_input"
    } == other_variable_cached
