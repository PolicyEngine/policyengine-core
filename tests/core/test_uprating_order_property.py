"""Properties of uprating: order independence and the reference rule.

Over the domain generated here, what a simulation returns for an uprated
variable, after earlier requests, equals byte for byte (dtype, shape and
bytes, so ``-0.0`` differs from ``0.0``) what a fresh simulation given the
same inputs returns for that period alone. It also equals the reference rule
in ``tests/fixtures/uprating_order.py``. Checked with and without
auto-carry-over. Regressions are in ``test_uprating_order.py``.

The domain:

* Inputs are stored in each variable's own unit. ``uprated_any_unit`` has no
  ``set_input`` helper, so its inputs are years or months.
* Earlier requests are plain calculations in either unit, ``ADD`` and
  ``DIVIDE``.
* The value may be read from the simulation, or from a branch or nested
  branch forked after those requests. The branch may set its own input for
  the target variable. The earlier requests then avoid the target period of
  the target variable: a value the parent calculated before the fork stays
  cached for the branch whatever its inputs are, which is not what this
  module tests.
* Targets are in the variable's definition unit.
"""

from __future__ import annotations

import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core import periods  # noqa: E402
from tests.fixtures.uprating_order import (  # noqa: E402
    COUNT,
    assert_bitwise_equal,
    build_system,
    reference,
    request,
    simulation,
)

SYSTEMS = {True: build_system(True), False: build_system(False)}

YEARS = [str(year) for year in range(2009, 2019)]
INPUT_YEARS = [str(year) for year in range(2010, 2017)]
MONTHS = [f"{year}-{month:02d}" for year in range(2011, 2016) for month in range(1, 13)]
INPUT_MONTHS = [month for month in MONTHS if "2011-06" <= month <= "2013-12"]
REQUEST_MONTHS = [month for month in MONTHS if "2011-11" <= month <= "2015-06"]

YEAR_VARIABLES = [
    "uprated",
    "uprated_count",
    "eligible",
    "uprated_if_eligible",
    "uprated_with_default",
    "uprated_any_unit",
]
ALL_VARIABLES = YEAR_VARIABLES + ["uprated_monthly"]
FLOATS = st.floats(min_value=-2000, max_value=2000, allow_nan=False, width=32)
VALUES = {variable: FLOATS for variable in ALL_VARIABLES}
VALUES["uprated_count"] = st.integers(min_value=-2000, max_value=2000)
VALUES["eligible"] = st.booleans()
INPUT_PERIODS = {variable: INPUT_YEARS for variable in YEAR_VARIABLES}
INPUT_PERIODS["uprated_monthly"] = INPUT_MONTHS
INPUT_PERIODS["uprated_any_unit"] = INPUT_YEARS + INPUT_MONTHS
# Other variables read ``eligible`` (``uprated_if_eligible`` is defined for
# it), so a parent request for them caches it at their period: no branch
# input for it.
BRANCH_INPUT_VARIABLES = [name for name in ALL_VARIABLES if name != "eligible"]


def _variable_inputs(variable, min_size=0):
    return st.dictionaries(
        st.sampled_from(INPUT_PERIODS[variable]),
        st.lists(VALUES[variable], min_size=COUNT, max_size=COUNT),
        min_size=min_size,
        max_size=3,
    )


def _requests_on(variable):
    """Requests for ``variable``: plain calculations in either unit (a yearly
    flow asked for a month caches a twelfth there, a monthly flow asked for a
    year caches the sum of its months), and the ADD and DIVIDE options."""
    years = st.sampled_from(YEARS)
    any_periods = st.sampled_from(YEARS + REQUEST_MONTHS)
    if variable == "uprated_monthly":
        return st.tuples(
            st.sampled_from(["calculate", "add"]), st.just(variable), any_periods
        )
    return st.one_of(
        st.tuples(st.just("calculate"), st.just(variable), any_periods),
        st.tuples(st.just("add"), st.just(variable), years),
        st.tuples(st.just("divide"), st.just(variable), any_periods),
    )


def _overlaps(first, second):
    first, second = periods.period(first), periods.period(second)
    return first.start <= second.stop and second.start <= first.stop


target_strategy = st.one_of(
    st.tuples(st.sampled_from(YEAR_VARIABLES), st.sampled_from(YEARS)),
    st.tuples(st.just("uprated_monthly"), st.sampled_from(REQUEST_MONTHS)),
)


@st.composite
def scenarios(draw):
    """A target, the inputs, the requests made before the target, the branch
    it is read from and that branch's own input, if any. The target variable
    always has an input, and about half the requests are for it: those are
    the requests that cache values it could be uprated from."""
    variable, period = draw(target_strategy)
    inputs = {
        name: draw(_variable_inputs(name, min_size=int(name == variable)))
        for name in ALL_VARIABLES
    }
    inputs = {name: values for name, values in inputs.items() if values}
    any_request = st.one_of(*(_requests_on(name) for name in ALL_VARIABLES))
    requests = draw(
        st.lists(st.one_of(_requests_on(variable), any_request), max_size=8)
    )
    branch = draw(st.sampled_from([None, "reform", "nested"]))
    branch_input = None
    if branch is not None and variable in BRANCH_INPUT_VARIABLES:
        branch_input = draw(
            st.none()
            | st.tuples(
                st.sampled_from(INPUT_PERIODS[variable]),
                st.lists(VALUES[variable], min_size=COUNT, max_size=COUNT),
            )
        )
    if branch_input is not None:
        requests = [
            (kind, name, requested)
            for kind, name, requested in requests
            if not (name == variable and _overlaps(requested, period))
        ]
    return inputs, requests, variable, period, branch, branch_input


def _read_from(built, branch, variable, branch_input):
    if branch == "reform":
        built = built.get_branch("reform")
    elif branch == "nested":
        built = built.get_branch("reform").get_branch("nested")
    if branch_input is not None:
        built.set_input(variable, branch_input[0], branch_input[1])
    return built


@settings(
    max_examples=500,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(scenario=scenarios(), auto_carry_over=st.booleans())
# The regressions in test_uprating_order.py, as examples of the property.
@example(
    scenario=(
        {"uprated_count": {"2012": [1001, 77]}},
        [
            ("calculate", "uprated_count", "2013"),
            ("calculate", "uprated_count", "2014"),
        ],
        "uprated_count",
        "2015",
        None,
        None,
    ),
    auto_carry_over=True,
)
@example(
    scenario=(
        {"uprated": {"2012": [1001.3, 77.7]}},
        [("calculate", "uprated", "2013"), ("divide", "uprated", "2014")],
        "uprated",
        "2015",
        "nested",
        None,
    ),
    auto_carry_over=False,
)
@example(
    scenario=(
        {
            "uprated_if_eligible": {"2012": [1000, 1000]},
            "eligible": {"2012": [True, True], "2013": [False, True]},
        },
        [("calculate", "uprated_if_eligible", "2013")],
        "uprated_if_eligible",
        "2015",
        "reform",
        None,
    ),
    auto_carry_over=True,
)
@example(
    scenario=(
        {
            "uprated_if_eligible": {"2012": [1000, 1000]},
            "eligible": {"2012": [True, True], "2013": [False, False]},
        },
        [("add", "uprated_if_eligible", "2013")],
        "uprated_if_eligible",
        "2014",
        None,
        None,
    ),
    auto_carry_over=True,
)
@example(
    scenario=(
        {"uprated_with_default": {"2016": [1, 2]}},
        [("calculate", "uprated_with_default", "2013")],
        "uprated_with_default",
        "2015",
        None,
        None,
    ),
    auto_carry_over=False,
)
@example(
    scenario=(
        {"uprated_any_unit": {"2012-03": [10, 20]}},
        [
            ("divide", "uprated_any_unit", "2012-03"),
            ("calculate", "uprated_any_unit", "2013"),
        ],
        "uprated_any_unit",
        "2015",
        None,
        None,
    ),
    auto_carry_over=True,
)
@example(
    scenario=(
        {"uprated_monthly": {"2011-06": [1001.3, 77.7]}},
        [
            ("calculate", "uprated_monthly", "2012-02"),
            ("add", "uprated_monthly", "2013"),
        ],
        "uprated_monthly",
        "2014-03",
        None,
        None,
    ),
    auto_carry_over=True,
)
# A branch input over a period the parent calculated before the fork.
@example(
    scenario=(
        {"uprated": {"2012": [1, 2]}},
        [("calculate", "uprated", "2013")],
        "uprated",
        "2015",
        "reform",
        ("2013", [300.0, 400.0]),
    ),
    auto_carry_over=True,
)
# Signed zero survives uprating from an input.
@example(
    scenario=(
        {"uprated": {"2012": [-0.0, 0.0]}},
        [("calculate", "uprated", "2013"), ("calculate", "uprated", "2014")],
        "uprated",
        "2015",
        None,
        None,
    ),
    auto_carry_over=False,
)
def test_uprating_depends_only_on_the_inputs(scenario, auto_carry_over):
    inputs, requests, variable, period, branch, branch_input = scenario
    system = SYSTEMS[auto_carry_over]

    built = simulation(system, inputs)
    for kind, requested_variable, requested_period in requests:
        request(built, kind, requested_variable, requested_period)
    result = _read_from(built, branch, variable, branch_input).calculate(
        variable, period
    )

    fresh = _read_from(simulation(system, inputs), branch, variable, branch_input)
    message = f"{variable} {period} after {requests}, branch {branch} {branch_input}"
    assert_bitwise_equal(result, fresh.calculate(variable, period), message)

    seen = {name: dict(values) for name, values in inputs.items()}
    if branch_input is not None:
        seen.setdefault(variable, {})[branch_input[0]] = branch_input[1]
    assert_bitwise_equal(result, reference(system, seen, variable, period), message)
