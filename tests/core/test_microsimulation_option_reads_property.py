"""Property: a Microsimulation's unweighted ADD and DIVIDE reads equal a Simulation's.

For any small dataset of households, people, weights and monthly and yearly
inputs, and any sequence of reads, a Microsimulation returns the same
unweighted values as a Simulation loaded from the same data, or raises the
same kind of error. The reads are formula results (each formula reads with
``options=[ADD]`` or ``options=[DIVIDE]``), the population calls formulas
make, ``calculate_add`` and ``calculate_divide`` with either value of
``decode_enums``, and plain reads at months, years and longer periods. Since
the Simulation ignores weights, weights never change an unweighted value.

Formula results also equal the same arithmetic done in NumPy, so the two
simulations cannot agree on a wrong value.

Enum variables are left out:
``test_enum_read_with_divide_in_a_microsimulation_matches_a_simulation``
records how their reads differ.
"""

import numpy as np
import pytest

from policyengine_core.model_api import ADD, DIVIDE, MONTH
from tests.fixtures.microsimulation_option_reads import (
    DATASET_YEAR,
    MONTHS,
    READERS,
    SOURCES,
    Inputs,
    expected,
    run,
    simulations,
    unweighted,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

# Whole amounts keep sums exact in float32, so a mismatch is never rounding.
AMOUNTS = st.integers(-10_000, 10_000).map(float)
WEIGHTS = st.floats(min_value=0, max_value=1e6, allow_nan=False)
PERIODS = ["2022", "2022-01", "2022-06", "2022-12", "2023", "month:2022-01:3"]
VARIABLES = sorted(variable.__name__ for variable in SOURCES + READERS)

REQUESTS = st.one_of(
    st.tuples(
        st.just("calculate"),
        st.sampled_from(VARIABLES),
        st.sampled_from(PERIODS),
    ),
    st.tuples(
        st.just("population"),
        st.sampled_from(["person", "household"]),
        st.sampled_from(VARIABLES),
        st.sampled_from(PERIODS),
        st.sampled_from([ADD, DIVIDE]),
    ),
    st.tuples(
        st.sampled_from(["add", "divide"]),
        st.sampled_from(VARIABLES),
        st.sampled_from(PERIODS),
        st.booleans(),
    ),
)


@st.composite
def inputs(draw):
    sizes = draw(st.lists(st.integers(1, 3), min_size=1, max_size=4))
    people, households = sum(sizes), len(sizes)

    def values(count):
        return tuple(draw(st.lists(AMOUNTS, min_size=count, max_size=count)))

    return Inputs(
        household_sizes=tuple(sizes),
        household_weight=tuple(
            draw(st.lists(WEIGHTS, min_size=households, max_size=households))
        ),
        monthly_pay=tuple(values(12) for _ in range(people)),
        yearly_grant=values(people),
        household_monthly_fee=tuple(values(12) for _ in range(households)),
        household_yearly_grant=values(households),
    )


@hypothesis.settings(max_examples=200, deadline=None)
@hypothesis.given(inputs(), st.lists(REQUESTS, min_size=1, max_size=6))
def test_unweighted_microsimulation_reads_equal_simulation_reads(data, requests):
    simulation, microsimulation = simulations(data)
    for request in requests:
        assert run(microsimulation, request) == run(simulation, request), request


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    inputs(),
    st.sampled_from(READERS),
    st.sampled_from(MONTHS),
)
def test_formula_reads_with_an_option_equal_numpy_arithmetic(data, reader, month):
    period = month if reader.definition_period == MONTH else DATASET_YEAR
    reference = expected(data, reader.__name__)
    for simulation in simulations(data):
        result = simulation.calculate(reader.__name__, period, **unweighted(simulation))
        np.testing.assert_allclose(
            np.asarray(result, dtype=float),
            np.asarray(reference, dtype=float),
            rtol=1e-6,
        )
