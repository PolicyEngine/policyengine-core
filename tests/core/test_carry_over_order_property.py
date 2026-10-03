"""Properties of auto-carry-over: order independence and the reference rule.

For any inputs and any earlier requests (plain calculations in either unit,
``ADD`` and ``DIVIDE``), what a simulation then returns for a period, read
from the simulation or from a branch or nested branch forked after those
requests, equals what a fresh simulation given the same inputs returns for
that period alone, and equals the reference rule in
``tests/fixtures/carry_over.py``. Regressions are in
``test_carry_over_order.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core import periods  # noqa: E402
from tests.fixtures.carry_over import (  # noqa: E402
    COUNT,
    alone,
    build_system,
    reference,
    request,
    simulation,
)

SYSTEM = build_system()

YEARS = [str(year) for year in range(2010, 2017)]
MONTHS = [f"{year}-{month:02d}" for year in (2012, 2013) for month in (1, 2, 6, 12)]
MULTI_YEAR = ["year:2012:2"]

YEAR_VARIABLES = [
    "carried",
    "carried_count",
    "eligible",
    "carried_if_eligible",
    "formula_until_2013",
    "year_input_without_helper",
]
MONTH_VARIABLES = ["carried_monthly", "month_input_without_helper"]
# What each variable's result reads: itself and its formula and defined_for.
READS = {
    "carried_if_eligible": ["carried_if_eligible", "eligible"],
    "formula_until_2013": ["formula_until_2013", "carried"],
}

FLOAT = st.floats(min_value=-100, max_value=100, allow_nan=False, width=32)
VALUES = {
    "carried_count": st.integers(min_value=-100, max_value=100),
    "eligible": st.booleans(),
}
INPUT_PERIODS = {variable: YEARS for variable in YEAR_VARIABLES}
INPUT_PERIODS["carried_monthly"] = MONTHS
# Variables with no set_input helper store inputs at any period.
INPUT_PERIODS["year_input_without_helper"] = YEARS + MONTHS + MULTI_YEAR
INPUT_PERIODS["month_input_without_helper"] = MONTHS + YEARS


def _shift(period, step):
    period = periods.period(period)
    return str(period.offset(step, period.unit))


@st.composite
def scenarios(draw):
    inputs = {}
    for variable, input_periods in INPUT_PERIODS.items():
        values = draw(
            st.dictionaries(
                st.sampled_from(input_periods),
                st.lists(VALUES.get(variable, FLOAT), min_size=COUNT, max_size=COUNT),
                max_size=3,
            )
        )
        if values:
            inputs[variable] = values
    variable = draw(st.sampled_from(YEAR_VARIABLES + MONTH_VARIABLES))
    unit_periods = YEARS if variable in YEAR_VARIABLES else MONTHS
    period = draw(st.sampled_from(unit_periods))
    # Bias requests toward what the target reads and toward periods next to
    # the inputs and the target: each bug needs a short chain of those.
    nearby = {period}
    for name in READS.get(variable, [variable]):
        nearby.update(inputs.get(name, {}))
    nearby.update(
        _shift(known, step)
        for known in list(nearby)
        if not known.startswith("year:")
        for step in (-1, 1)
    )
    nearby = sorted(nearby)
    request_variables = st.one_of(
        st.sampled_from(READS.get(variable, [variable])),
        st.sampled_from(YEAR_VARIABLES + MONTH_VARIABLES),
    )

    @st.composite
    def one_request(draw):
        name = draw(request_variables)
        kind = draw(st.sampled_from(["calculate", "calculate", "add", "divide"]))
        candidates = nearby + YEARS + MONTHS
        if name in YEAR_VARIABLES:
            candidates += MULTI_YEAR
        requested = draw(st.sampled_from(candidates))
        unit = periods.period(requested).unit
        if kind == "divide" and (
            name not in YEAR_VARIABLES or periods.period(requested).size != 1
        ):
            kind = "calculate"
        if name in MONTH_VARIABLES and requested.startswith("year:"):
            kind = "add"
        if kind == "add" and name in YEAR_VARIABLES and unit == periods.MONTH:
            kind = "calculate"
        if kind == "calculate" and requested.startswith("year:"):
            kind = "add"
        return kind, name, requested

    requests = draw(st.lists(one_request(), max_size=8))
    branch = draw(st.sampled_from([None, "reform", "nested"]))
    branch_requests = draw(st.lists(one_request(), max_size=4)) if branch else []
    return inputs, requests, branch, branch_requests, (variable, period)


@settings(
    max_examples=400,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(scenario=scenarios())
# One example per regression shape in test_carry_over_order.py.
@example(
    scenario=(
        {"carried": {"2012": [7, 8]}},
        [("calculate", "carried", "2014")],
        None,
        [],
        ("carried", "2013"),
    )
)
@example(
    scenario=(
        {
            "carried_if_eligible": {"2012": [7, 8]},
            "eligible": {"2013": [False, True], "2014": [True, True]},
        },
        [("calculate", "carried_if_eligible", "2013")],
        None,
        [],
        ("carried_if_eligible", "2014"),
    )
)
@example(
    scenario=(
        {
            "carried_if_eligible": {"2012": [7, 8]},
            "eligible": {"2013": [False, False], "2014": [True, True]},
        },
        [("add", "carried_if_eligible", "2013")],
        "reform",
        [],
        ("carried_if_eligible", "2014"),
    )
)
@example(
    scenario=(
        {"carried": {"2012": [1, 2]}},
        [("calculate", "formula_until_2013", "2013")],
        None,
        [],
        ("formula_until_2013", "2014"),
    )
)
@example(
    scenario=(
        {"year_input_without_helper": {"year:2012:2": [120, 24]}},
        [("add", "year_input_without_helper", "year:2012:2")],
        None,
        [],
        ("year_input_without_helper", "2014"),
    )
)
@example(
    scenario=(
        {"year_input_without_helper": {"2012-01": [120, 24]}},
        [("divide", "year_input_without_helper", "2012-01")],
        "nested",
        [],
        ("year_input_without_helper", "2014"),
    )
)
@example(
    scenario=(
        {"month_input_without_helper": {"2013": [120, 24]}},
        [("add", "month_input_without_helper", "2013")],
        None,
        [],
        ("month_input_without_helper", "2013-06"),
    )
)
@example(
    scenario=(
        {"year_input_without_helper": {"2013-01": [2, 2], "2013": [1, 1]}},
        [],
        None,
        [],
        ("year_input_without_helper", "2014"),
    )
)
@example(
    scenario=(
        {"year_input_without_helper": {"2013-01": [3, 4]}},
        [],
        None,
        [],
        ("year_input_without_helper", "2013"),
    )
)
@example(
    scenario=(
        {"year_input_without_helper": {"year:2012:2": [120, 24]}},
        [],
        "reform",
        [("add", "year_input_without_helper", "year:2012:2")],
        ("year_input_without_helper", "2014"),
    )
)
def test_carry_over_depends_only_on_the_inputs(scenario):
    inputs, requests, branch, branch_requests, (variable, period) = scenario
    built = simulation(SYSTEM, inputs)
    for kind, name, requested in requests:
        request(built, kind, name, requested)
    if branch == "reform":
        built = built.get_branch("reform")
    elif branch == "nested":
        built = built.get_branch("reform").get_branch("nested")
    for kind, name, requested in branch_requests:
        request(built, kind, name, requested)
    result = built.calculate(variable, period)

    message = (
        f"{variable} {period} after {requests}, then {branch_requests} on {branch}"
    )
    np.testing.assert_array_equal(
        result, alone(SYSTEM, inputs, variable, period), err_msg=message
    )
    np.testing.assert_array_equal(
        result, reference(SYSTEM, inputs, variable, period), err_msg=message
    )


# ----- The order of period ends ----- #

from datetime import date  # noqa: E402

from policyengine_core.simulations.simulation import _end_order  # noqa: E402

_starts = st.dates(min_value=date(1, 1, 1), max_value=date(9999, 12, 31))
_units = st.sampled_from([periods.DAY, periods.MONTH, periods.YEAR])


def _period(unit, start, size):
    return periods.Period(
        (unit, periods.Instant((start.year, start.month, start.day)), size)
    )


def _numpy_day_after(period):
    """The day after the period's last day, on numpy's calendar, which runs
    past year 9999: an implementation that shares nothing with
    ``_end_order`` or ``Period.stop``."""
    unit, (year, month, day), size = period
    if unit == periods.DAY:
        return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}") + np.timedelta64(
            size, "D"
        )
    months = size if unit == periods.MONTH else 12 * size
    first = np.datetime64(f"{year:04d}-{month:02d}") + np.timedelta64(months, "M")
    return first.astype("datetime64[D]") + np.timedelta64(day - 1, "D")


@settings(max_examples=1000, deadline=None, derandomize=True)
@given(unit=_units, start=_starts, size=st.integers(min_value=1, max_value=5_000_000))
@example(unit=periods.DAY, start=date(9999, 12, 30), size=3)
@example(unit=periods.MONTH, start=date(2012, 1, 31), size=1)
@example(unit=periods.YEAR, start=date(2012, 2, 29), size=1)
@example(unit=periods.MONTH, start=date(9999, 2, 1), size=24)
def test_end_order_is_the_day_after_the_last_day(unit, start, size):
    """Against two other implementations: ``Period.stop`` where it has a
    value, and numpy's calendar everywhere."""
    period = _period(unit, start, size)
    day_one = np.datetime64("0001-01-01")
    expected = int((_numpy_day_after(period) - day_one) / np.timedelta64(1, "D")) + 1
    assert _end_order(period) == expected
    try:
        stop = date(*period.stop)
    except (OverflowError, ValueError):
        # Ends after 9999-12-31, the last date ``datetime`` has: ``stop``
        # raises, or gives a year no ``date`` can hold.
        assert expected > date.max.toordinal() + 1
    else:
        assert _end_order(period) == stop.toordinal() + 1


@settings(max_examples=500, deadline=None, derandomize=True)
@given(
    unit=_units,
    start=_starts,
    size=st.integers(min_value=1, max_value=5_000_000),
    more=st.integers(min_value=1, max_value=1000),
)
def test_a_longer_period_from_the_same_day_ends_later(unit, start, size, more):
    assert _end_order(_period(unit, start, size + more)) > _end_order(
        _period(unit, start, size)
    )


def test_eternity_ends_after_every_other_period():
    eternity = periods.period(periods.ETERNITY)
    assert _end_order(eternity) > _end_order(
        _period(periods.YEAR, date(9999, 1, 1), 5_000_000)
    )
