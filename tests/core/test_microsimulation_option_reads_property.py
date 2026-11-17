"""Property: a Microsimulation's unweighted ADD and DIVIDE reads equal a Simulation's.

For any small dataset of households, people listed in any order, weights and
monthly and yearly inputs, and any sequence of reads, a Microsimulation
returns the same unweighted values as a Simulation loaded from the same data,
or raises the same kind of error. The reads are:

- formula results: each formula reads with ``options=[ADD]`` or
  ``options=[DIVIDE]``, and one projects such a read to members;
- the population calls formulas make;
- ``calculate_add`` and ``calculate_divide``, with either value of
  ``decode_enums``;
- plain reads at months, years and longer periods.

Population calls and the two methods read each variable over periods its
option accepts. Since the Simulation ignores weights, weights never change an
unweighted value.

The population calls return writable NumPy arrays in both classes. Formula
results also equal the same arithmetic done in NumPy, so the two simulations
cannot agree on a wrong value.

Enum variables are left out:
``test_enum_read_with_divide_over_a_year_gives_member_names_in_a_microsimulation``
records how their reads differ.
"""

import numpy as np
import pytest

from policyengine_core.model_api import ADD, DIVIDE, MONTH, YEAR
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

# Whole monthly amounts, and yearly amounts in multiples of 12, keep every sum
# and monthly share exact in float32, so a mismatch is never rounding.
AMOUNTS = st.integers(-10_000, 10_000).map(float)
YEARLY_AMOUNTS = st.integers(-10_000, 10_000).map(lambda share: 12.0 * share)
WEIGHTS = st.floats(min_value=0, max_value=1e6, allow_nan=False)
PERIODS = ["2022", "2022-01", "2022-06", "2022-12", "2023", "month:2022-01:3"]
VARIABLES = SOURCES + READERS

# The periods each option reads a variable over, by the variable's definition
# period. Over any other period the read raises in both classes.
OPTION_PERIODS = {
    (ADD, MONTH): ["2022", "2022-01", "2022-06", "2023", "month:2022-01:3"],
    (ADD, YEAR): ["2022", "2023"],
    (DIVIDE, YEAR): ["2022", "2022-01", "2022-06", "2022-12", "2023"],
}


def option_reads(variable) -> list:
    """Each option and period that reads ``variable`` without an error."""
    return [
        (option, period)
        for (option, definition_period), periods in OPTION_PERIODS.items()
        if definition_period == variable.definition_period
        for period in periods
    ]


def readable(entity_key: str, variable) -> bool:
    # A household's population sums its members' person variables.
    return variable.entity.key == entity_key or (
        entity_key == "household" and variable.entity.is_person
    )


POPULATION_CALLS = [
    ("population", entity_key, variable.__name__, period, option)
    for entity_key in ["person", "household"]
    for variable in VARIABLES
    if readable(entity_key, variable)
    for option, period in option_reads(variable)
]
METHOD_CALLS = [
    ({ADD: "add", DIVIDE: "divide"}[option], variable.__name__, period, decode)
    for variable in VARIABLES
    for option, period in option_reads(variable)
    for decode in [False, True]
]
REQUESTS = st.one_of(
    st.tuples(
        st.just("calculate"),
        st.sampled_from(sorted(variable.__name__ for variable in VARIABLES)),
        st.sampled_from(PERIODS),
    ),
    st.sampled_from(POPULATION_CALLS),
    st.sampled_from(METHOD_CALLS),
)


@st.composite
def inputs(draw):
    sizes = draw(st.lists(st.integers(1, 3), min_size=1, max_size=4))
    households = len(sizes)
    # The data can list a household's members anywhere.
    household_of_person = draw(
        st.permutations(
            [household for household, size in enumerate(sizes) for _ in range(size)]
        )
    )
    people = len(household_of_person)

    def values(count, amounts=AMOUNTS):
        return tuple(draw(st.lists(amounts, min_size=count, max_size=count)))

    return Inputs(
        household_of_person=tuple(household_of_person),
        household_weight=tuple(
            draw(st.lists(WEIGHTS, min_size=households, max_size=households))
        ),
        monthly_pay=tuple(values(12) for _ in range(people)),
        yearly_grant=values(people, YEARLY_AMOUNTS),
        household_monthly_fee=tuple(values(12) for _ in range(households)),
        household_yearly_grant=values(households, YEARLY_AMOUNTS),
    )


def outcome_kind(outcome) -> str:
    """Whether a read raised, returned only zeros, or returned other values."""
    if isinstance(outcome, str):
        return "an error"
    return "values" if np.any(np.asarray(outcome, dtype=float)) else "zeros"


@hypothesis.settings(max_examples=200, deadline=None)
@hypothesis.given(inputs(), st.lists(REQUESTS, min_size=1, max_size=6))
def test_unweighted_microsimulation_reads_equal_simulation_reads(data, requests):
    simulation, microsimulation = simulations(data)
    for request in requests:
        result = run(microsimulation, request)
        hypothesis.event(f"{request[0]} read gives {outcome_kind(result)}")
        assert result == run(simulation, request), request


@hypothesis.settings(max_examples=50, deadline=None)
@hypothesis.given(inputs(), st.sampled_from(POPULATION_CALLS))
def test_population_reads_with_an_option_return_numpy_arrays(data, call):
    # pandas pairs two Series' values by index label, and a projected Series
    # carries each member's group position, so formulas get positional arrays.
    # Formulas may also write to what they read.
    _, entity_key, variable, period, option = call
    for simulation in simulations(data):
        read = simulation.populations[entity_key](variable, period, options=[option])
        assert type(read) is np.ndarray, (type(simulation).__name__, call)
        assert read.flags.writeable, (type(simulation).__name__, call)


@pytest.mark.parametrize("reader", READERS, ids=lambda reader: reader.__name__)
@hypothesis.settings(max_examples=25, deadline=None)
@hypothesis.given(inputs(), st.sampled_from(MONTHS))
def test_formula_reads_with_an_option_equal_numpy_arithmetic(reader, data, month):
    period = month if reader.definition_period == MONTH else DATASET_YEAR
    reference = expected(data, reader.__name__)
    for simulation in simulations(data):
        result = simulation.calculate(reader.__name__, period, **unweighted(simulation))
        np.testing.assert_array_equal(
            np.asarray(result, dtype=float), np.asarray(reference, dtype=float)
        )
