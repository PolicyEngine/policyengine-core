"""Property: running ADD or DIVIDE first never changes a later result.

For a monthly or yearly variable of any quantity type and value type, with
or without a formula, and with inputs stored at any periods, every request
(plain ``calculate``, ``calculate_add`` or ``calculate_divide`` at a single
or multi-period period) returns the same values, or raises the same kind of
error, whatever requests ran before it. The reference is the same request in
a new simulation with the same inputs.

Auto-carry-over is off here: on its own, carrying over calculated values is
a separate order dependence (policyengine-core#562).
"""

import pytest

from tests.fixtures.option_caches import (
    FLOW,
    MONTH,
    STOCK,
    YEAR,
    build,
    make_probe,
    run,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

OPERATIONS = ["calculate", "add", "divide"]
PERIODS = [
    "2012",
    "2013",
    "2012-01",
    "2012-06",
    "2012-12",
    "2013-01",
    "year:2012:2",
    "month:2012-01:3",
]
INPUT_PERIODS = ["2012", "2013", "2012-01", "2012-06", "2012-12"]
REQUESTS = st.tuples(st.sampled_from(OPERATIONS), st.sampled_from(PERIODS))


@st.composite
def scenarios(draw):
    probe = dict(
        definition_period=draw(st.sampled_from([MONTH, YEAR])),
        quantity_type=draw(st.sampled_from([STOCK, FLOW])),
        value_type=draw(st.sampled_from([float, int, bool])),
        with_formula=draw(st.booleans()),
        set_input=draw(st.sampled_from(["default", None])),
    )
    inputs = draw(
        st.dictionaries(st.sampled_from(INPUT_PERIODS), st.integers(0, 40), max_size=3)
    )
    prior = draw(st.lists(REQUESTS, max_size=5))
    target = draw(REQUESTS)
    return probe, inputs, prior, target


def build_or_none(probe, inputs):
    try:
        return build(probe, inputs)
    except (TypeError, ValueError):
        # Inputs this variable cannot take at those periods (spreading a
        # boolean over months, for instance, subtracts booleans).
        return None


@hypothesis.settings(max_examples=400, deadline=None)
@hypothesis.given(scenarios())
def test_earlier_requests_do_not_change_a_result(scenario):
    attributes, inputs, prior, target = scenario
    probe = make_probe(**attributes)
    fresh = build_or_none(probe, inputs)
    hypothesis.assume(fresh is not None)
    used = build(probe, inputs)
    for request in prior:
        run(used, *request)
    assert run(used, *target) == run(fresh, *target)
