"""Finite anchored recurrences agree with an independent iterative evaluator.

Every dependency moves one or two years in the same direction. Backward
formulas start in 2010; forward formulas end in 2024. This gives the oracle
a finite temporal order without using the simulation's recursion scheduler,
cache, tracer, or exception handling. The existing broader order property
continues to cover mixed graphs and unanchored recurrences.
"""

from typing import NamedTuple

import numpy as np
import pytest

from policyengine_core.country_template.entities import Person
from policyengine_core.periods import YEAR, period
from policyengine_core.variables import Variable
from tests.fixtures.spirals import build, make_recurrence

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

FIRST_YEAR = 2010
LAST_YEAR = 2024


class Recurrence(NamedTuple):
    name: str
    terms: tuple
    guarded: bool


class Scenario(NamedTuple):
    recurrences: tuple
    direction: int
    inputs: dict
    loops: int
    prior: tuple
    target: tuple


@st.composite
def anchored_scenarios(draw):
    names = ("r0", "r1")[: draw(st.integers(1, 2))]
    direction = draw(st.sampled_from((-1, 1)))
    recurrences = []
    for name in names:
        terms = draw(
            st.lists(
                st.tuples(st.sampled_from(names), st.integers(1, 2)),
                min_size=1,
                max_size=2,
            )
        )
        recurrences.append(
            Recurrence(
                name,
                tuple((other, direction * distance) for other, distance in terms),
                draw(st.booleans()),
            )
        )
    requests = st.tuples(st.sampled_from(names), st.integers(FIRST_YEAR, LAST_YEAR))
    inputs = draw(
        st.dictionaries(
            requests,
            st.floats(
                min_value=-9,
                max_value=9,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            max_size=3,
        )
    )
    return Scenario(
        tuple(recurrences),
        direction,
        inputs,
        draw(st.sampled_from((1, 2, 3, 10))),
        tuple(draw(st.lists(requests, max_size=3))),
        draw(requests),
    )


def iterative_values(scenario):
    """Fill the recurrence table in dependency order, including its boundary.

    ``filled_array(1.0)`` uses NumPy's inferred float64 dtype. Each formula
    adds its already-cast float32 dependencies in order and then casts its
    completed result to the float variable's float32 dtype. Casting only
    the final mathematical answer would miss rounding at earlier years.
    """
    values = {}
    years = range(FIRST_YEAR - 2, LAST_YEAR + 3)
    if scenario.direction > 0:
        years = reversed(years)
    for year in years:
        outside_formula = (
            year < FIRST_YEAR if scenario.direction < 0 else year > LAST_YEAR
        )
        for recurrence in scenario.recurrences:
            key = (recurrence.name, year)
            if key in scenario.inputs:
                value = np.array([scenario.inputs[key]], dtype=np.float32)
            elif outside_formula:
                value = np.zeros(1, dtype=np.float32)
            else:
                value = np.full(1, 1.0)
                for other, offset in recurrence.terms:
                    value = value + values[(other, year + offset)]
                value = value.astype(np.float32)
            values[key] = value
    return values


def recurrence_variable(recurrence, direction, requires_caller):
    def formula(person, period):
        value = person.filled_array(1.0)
        for other, offset in recurrence.terms:
            value = value + person(other, period.offset(offset, "year"))
        return value

    def guarded_formula(person, period):
        try:
            return formula(person, period)
        except:  # noqa: E722 - exercises a swallowed scheduling request
            return person.filled_array(-1000.0)

    attributes = dict(
        value_type=float,
        entity=Person,
        definition_period=YEAR,
        label="Finite anchored recurrence",
    )
    formula_name = f"formula_{FIRST_YEAR}" if direction < 0 else "formula"
    attributes[formula_name] = guarded_formula if recurrence.guarded else formula
    if direction > 0:
        attributes["end"] = f"{LAST_YEAR}-12-31"
    if requires_caller:
        attributes["requires_computation_after"] = "caller"
    return type(recurrence.name, (Variable,), attributes)


def new_simulation(scenario, traced, storage, requires_caller=False):
    variables = [
        recurrence_variable(recurrence, scenario.direction, requires_caller)
        for recurrence in scenario.recurrences
    ]
    if requires_caller:

        class caller(Variable):
            value_type = float
            entity = Person
            definition_period = YEAR
            label = "Authorizes the recursive calculation"

            def formula(person, period):
                return person("r0", period)

        variables.append(caller)
    simulation = build(variables, scenario.inputs, scenario.loops)
    simulation.tax_benefit_system.auto_carry_over_input_variables = False
    simulation.trace = traced
    names = [variable.__name__ for variable in variables]
    for name in names:
        assert simulation.tax_benefit_system.get_variable(name).dtype == np.float32
    if storage == "drop":
        for name in names:
            simulation.get_holder(name)._do_not_store = True
    elif storage == "blacklist":
        simulation.opt_out_cache = True
        simulation.tax_benefit_system.cache_blacklist = set(names)
    return simulation


def assert_request(simulation, name, year, expected, traced):
    actual = simulation.calculate(name, str(year))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if traced:
        trace = simulation.tracer.get_serialized_flat_trace()
        assert trace[f"{name}<{year}, (default)>"]["value"] == expected.tolist()
    assert simulation.tracer.stack == []


GUARDED_BACKWARD = Scenario(
    (Recurrence("r0", (("r0", -1),), True),),
    -1,
    {},
    1,
    (("r0", 2023),),
    ("r0", 2024),
)
GUARDED_FORWARD = Scenario(
    (Recurrence("r0", (("r0", 1),), True),),
    1,
    {("r0", 2024): 0.25},
    1,
    (("r0", 2011),),
    ("r0", 2010),
)


@hypothesis.settings(max_examples=60, deadline=None, derandomize=True)
@hypothesis.given(
    scenario=anchored_scenarios(),
    traced=st.booleans(),
    storage=st.sampled_from(("store", "drop", "blacklist")),
)
@hypothesis.example(scenario=GUARDED_BACKWARD, traced=True, storage="drop")
@hypothesis.example(scenario=GUARDED_FORWARD, traced=False, storage="blacklist")
def test_anchored_scheduler_matches_iterative_values(scenario, traced, storage):
    expected = iterative_values(scenario)
    used = new_simulation(scenario, traced, storage)
    for name, year in scenario.prior:
        assert_request(used, name, year, expected[(name, year)], traced)
    name, year = scenario.target
    assert_request(used, name, year, expected[(name, year)], traced)
    # Compare with both tracing modes on a fresh simulation. Neither the
    # stored work nor tracing may change the independently evaluated value.
    fresh = new_simulation(scenario, not traced, storage)
    assert_request(fresh, name, year, expected[(name, year)], not traced)


@hypothesis.settings(max_examples=40, deadline=None, derandomize=True)
@hypothesis.given(
    scenario=anchored_scenarios(),
    traced=st.booleans(),
    storage=st.sampled_from(("store", "drop", "blacklist")),
)
@hypothesis.example(scenario=GUARDED_BACKWARD, traced=True, storage="drop")
def test_deferred_periods_keep_caller_authorization(scenario, traced, storage):
    expected = iterative_values(scenario)
    used = new_simulation(scenario, traced, storage, requires_caller=True)
    for _, year in scenario.prior:
        assert_request(used, "caller", year, expected[("r0", year)], traced)
    _, year = scenario.target
    assert_request(used, "caller", year, expected[("r0", year)], traced)
    fresh = new_simulation(scenario, not traced, storage, requires_caller=True)
    assert_request(fresh, "caller", year, expected[("r0", year)], not traced)


@hypothesis.settings(max_examples=24, deadline=None, derandomize=True)
@hypothesis.given(
    input_year=st.integers(2010, 2014),
    loops=st.sampled_from((1, 2, 3, 10)),
    distance=st.integers(1, 5),
    traced=st.booleans(),
    removal=st.sampled_from(("period", "all", "invalidate", "derived_replacement")),
)
def test_scheduler_uses_current_input_provenance(
    input_year, loops, distance, traced, removal
):
    error_year = input_year + 1

    class recursive(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Undated recurrence with an unreachable error"

        def formula(person, period):
            if period.start.year == error_year:
                raise ValueError("This period is beyond the unanchored depth limit")
            return person("recursive", period.last_year) + 1

    used = build([recursive], {("recursive", input_year): 40}, loops)
    used.trace = traced
    input_period = period(str(input_year))
    if removal == "all":
        used.delete_arrays("recursive")
    elif removal == "invalidate":
        used.invalidate_cache_entry("recursive", input_period)
        used.purge_cache_of_invalid_values()
    else:
        used.delete_arrays("recursive", input_period)
        if removal == "derived_replacement":
            used.get_holder("recursive").put_in_cache([7], input_period, derived=True)
    assert used.get_holder("recursive").get_input_periods() == []

    # Every generated request stops above the error in an unanchored
    # simulation. Neither deleted input history nor a derived replacement
    # may authorize the scheduler to reach it.
    target = error_year + loops + distance
    fresh = build([recursive], max_spiral_loops=loops)
    fresh.trace = not traced
    expected = fresh.calculate("recursive", str(target))
    np.testing.assert_array_equal(expected, np.array([loops], dtype=np.float32))
    assert_request(used, "recursive", target, expected, traced)


@hypothesis.settings(max_examples=24, deadline=None, derandomize=True)
@hypothesis.given(
    loops=st.sampled_from((1, 2, 3, 10)),
    year=st.integers(FIRST_YEAR, LAST_YEAR),
    later=st.integers(1, 3),
    traced=st.booleans(),
    branch=st.booleans(),
    input_value=st.integers(50, 150),
)
def test_cut_copies_and_descendant_inputs_are_independent(
    loops, year, later, traced, branch, input_value
):
    copies = []

    class wrapper(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Copies the simulation while a cut is pending"

        def formula(person, period):
            value = person("unanchored", period)
            simulation = person.simulation
            copies.append(
                simulation.get_branch("copy") if branch else simulation.clone()
            )
            return value

    recursive = make_recurrence("unanchored", (("unanchored", 1),))
    simulation = build([recursive, wrapper], max_spiral_loops=loops)
    simulation.trace = traced
    descendant = simulation.get_branch("one").get_branch("default")
    descendant.set_input("wrapper", str(year), [input_value])
    expected = np.array([loops], dtype=np.float32)
    assert_request(simulation, "wrapper", year, expected, traced)
    assert descendant.get_array("wrapper", str(year)).tolist() == [input_value]
    assert descendant.calculate("wrapper", str(year)).tolist() == [input_value]
    copy = copies[0]
    actual = copy.calculate("unanchored", str(year + later))
    np.testing.assert_array_equal(actual, expected)
    fresh = build([recursive], max_spiral_loops=loops)
    np.testing.assert_array_equal(
        actual, fresh.calculate("unanchored", str(year + later))
    )
