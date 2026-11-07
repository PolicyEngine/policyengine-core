"""A reform simulation's baseline is what the builder makes of the situation.

For any reform made of changes to input variables' defaults (person and
household variables; float, int, enum, date and string; monthly variables
whose yearly inputs are divided or dispatched between months, and an eternal
one), neutralized variables (inputs and formulas), a variable only the reform
has, and a changed tax rate; and for any situation of one to three people in
one or two households that gives any of those variables, for any of several
periods, to some people or households and not to others, with no axes, one
dimension of parallel axes, or two perpendicular dimensions (or that gives
each variable's values for everyone, without entities):

* Differential: the baseline calculates, for each output and period, what the
  simulation ``SimulationBuilder`` makes of the situation under the baseline
  system alone calculates. The variable only the reform has is left out of
  the situation first: the baseline system cannot take it.
* The baseline has the same population and records the same inputs
  (``_user_input_keys``, ``input_variables``) as that simulation, and exports
  the same ``to_input_dict()``.
* It is the reform simulation's ``"baseline"`` branch, under the baseline
  system.
* The reform simulation calculates what the builder makes of the situation
  under another copy of the reform's system: the fix leaves it as it was.

Examples, including the cases that failed before (PolicyEngine/
policyengine-core#612), are in ``test_reform_baseline_situation.py``.
"""

from __future__ import annotations

import copy
import datetime
import functools

import numpy as np
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import Simulation  # noqa: E402
from policyengine_core.periods import instant  # noqa: E402
from policyengine_core.reforms import Reform  # noqa: E402
from policyengine_core.simulations import SimulationBuilder  # noqa: E402
from policyengine_core.variables import Variable  # noqa: E402
from tests.core.test_reform_baseline_system import doubled_salary  # noqa: E402

PERIOD = "2022-01"
SYSTEM = Simulation.default_tax_benefit_system_instance
ADDED = "doubled_salary"
MONTHS = ["2022-01", "2022-03"]
INPUT_PERIODS = [*MONTHS, "2022"]
STATUSES = ["owner", "tenant", "free_lodger", "homeless"]
DATES = st.dates(datetime.date(1930, 1, 1), datetime.date(2021, 12, 31))
POSTAL_CODES = st.text(alphabet="0123456789", min_size=5, max_size=5)

# Each input variable: the values a situation (or a reform's default) gives
# it, and the periods a situation gives it for.
PERSON_INPUTS = {
    "salary": (st.integers(0, 10_000).map(float), INPUT_PERIODS),
    "age": (st.integers(0, 100), INPUT_PERIODS),
    "birth": (DATES.map(str), ["ETERNITY"]),
}
HOUSEHOLD_INPUTS = {
    "rent": (st.integers(0, 2000).map(float), INPUT_PERIODS),
    "accommodation_size": (st.integers(0, 300).map(float), INPUT_PERIODS),
    "housing_occupancy_status": (st.sampled_from(STATUSES), INPUT_PERIODS),
    "postal_code": (POSTAL_CODES, INPUT_PERIODS),
}
DEFAULTS = {
    "salary": st.integers(1, 5000).map(float),
    "age": st.integers(1, 90),
    "birth": DATES,
    "rent": st.integers(1, 2000).map(float),
    "accommodation_size": st.integers(1, 300).map(float),
    "housing_occupancy_status": st.sampled_from(STATUSES),
    "postal_code": POSTAL_CODES,
}
NEUTRALIZABLE = ["salary", "rent", "income_tax", "housing_tax"]
AXES = {
    "person": ["salary", "age"],
    "household": ["rent", "accommodation_size"],
}
# What is compared: each variable at the periods its definition takes.
OUTPUT_MONTHS = [*MONTHS, "2022-07"]
OUTPUTS = {
    "salary": OUTPUT_MONTHS,
    "age": OUTPUT_MONTHS,
    "birth": [PERIOD],
    "income_tax": OUTPUT_MONTHS,
    "social_security_contribution": OUTPUT_MONTHS,
    "disposable_income": OUTPUT_MONTHS,
    "rent": OUTPUT_MONTHS,
    "accommodation_size": OUTPUT_MONTHS,
    "housing_occupancy_status": OUTPUT_MONTHS,
    "postal_code": OUTPUT_MONTHS,
    "household_income": OUTPUT_MONTHS,
    "total_taxes": OUTPUT_MONTHS,
    "housing_tax": ["2022"],
}


@functools.lru_cache(maxsize=None)
def _reform(spec: tuple) -> type:
    """A reform class from ``spec``: (defaults, neutralized, added, rate)."""
    defaults, neutralized, added, rate = spec

    def apply(self):
        for name, value in defaults:
            if name == "housing_occupancy_status":
                # A member of this system's own enum class.
                value = self.get_variable(name).possible_values[value]
            self.update_variable(type(name, (Variable,), {"default_value": value}))
        for name in neutralized:
            self.neutralize_variable(name)
        if added:
            self.update_variable(doubled_salary)
        if rate is not None:

            def modify(parameters):
                parameters.taxes.income_tax_rate.update(
                    start=instant("2015-01-01"), value=rate
                )
                return parameters

            self.modify_parameters(modify)

    return type("generated_reform", (Reform,), {"apply": apply})


@st.composite
def reform_specs(draw):
    names = draw(st.lists(st.sampled_from(sorted(DEFAULTS)), unique=True))
    defaults = tuple((name, draw(DEFAULTS[name])) for name in names)
    neutralized = tuple(draw(st.lists(st.sampled_from(NEUTRALIZABLE), unique=True)))
    added = draw(st.booleans())
    rate = draw(st.none() | st.sampled_from([0.0, 0.3]))
    return defaults, neutralized, added, rate


def _given(draw, inputs, extra=None):
    """Variable values a person or household is given: some, none or all."""
    inputs = {**inputs, **(extra or {})}
    entry = {}
    for name in sorted(inputs):
        values, periods = inputs[name]
        dated = draw(st.dictionaries(st.sampled_from(periods), values, max_size=2))
        if dated:
            entry[name] = dated
    return entry


@st.composite
def axes(draw, counts):
    dimensions = draw(st.sampled_from([0, 0, 1, 2]))
    result = []
    for _ in range(dimensions):
        entity = draw(st.sampled_from(sorted(AXES)))
        count = draw(st.integers(1, 3))
        parallel = []
        for _ in range(draw(st.integers(1, 2))):
            low = draw(st.integers(0, 1000))
            parallel.append(
                {
                    "name": draw(st.sampled_from(AXES[entity])),
                    "count": count,
                    "index": draw(st.integers(0, counts[entity] - 1)),
                    "min": low,
                    "max": low + draw(st.integers(0, 5000)),
                    "period": draw(st.sampled_from([PERIOD, "2022"])),
                }
            )
        result.append(parallel)
    return result


@st.composite
def situations(draw):
    people = [f"p{index}" for index in range(draw(st.integers(1, 3)))]
    household_count = draw(st.integers(1, min(2, len(people))))
    # The first people found the households; the rest join any of them.
    homes = list(range(household_count)) + draw(
        st.lists(
            st.integers(0, household_count - 1),
            min_size=len(people) - household_count,
            max_size=len(people) - household_count,
        )
    )
    added = {ADDED: (st.integers(0, 1000).map(float), MONTHS)}
    situation = {
        "persons": {person: _given(draw, PERSON_INPUTS, added) for person in people},
        "households": {},
    }
    for home in range(household_count):
        members = [person for person, at in zip(people, homes) if at == home]
        household = {"parents": members[:2], **_given(draw, HOUSEHOLD_INPUTS)}
        if members[2:]:
            household["children"] = members[2:]
        situation["households"][f"h{home}"] = household
    dimensions = draw(axes({"person": len(people), "household": household_count}))
    if dimensions:
        situation["axes"] = dimensions
    return situation


@st.composite
def situations_without_entities(draw):
    """Variables' values for each of one to three people, each in a household.

    The builder takes the number of people from the first value.
    """
    count = draw(st.integers(1, 3))
    inputs = {
        **PERSON_INPUTS,
        **HOUSEHOLD_INPUTS,
        ADDED: (st.integers(0, 1000).map(float), MONTHS),
    }
    situation = {}
    for name in draw(st.permutations(sorted(inputs))):
        values, periods = inputs[name]
        dated = draw(
            st.dictionaries(
                st.sampled_from(periods),
                st.lists(values, min_size=count, max_size=count),
                max_size=2,
            )
        )
        if dated:
            situation[name] = dated
    if not set(situation) - {ADDED}:
        # Left without the added variable, the situation keeps its people.
        situation["salary"] = {PERIOD: [0.0] * count}
    return situation


def _built(situation, system):
    builder = SimulationBuilder()
    builder.default_period = PERIOD
    return builder.build_from_dict(system, copy.deepcopy(situation))


def _without_added(situation):
    stripped = copy.deepcopy(situation)
    stripped.pop(ADDED, None)
    for person in stripped.get("persons", {}).values():
        person.pop(ADDED, None)
    return stripped


def _assert_calculates_the_same(simulation, expected):
    for variable, periods in OUTPUTS.items():
        for period in periods:
            np.testing.assert_array_equal(
                np.asarray(simulation.calculate(variable, period)),
                np.asarray(expected.calculate(variable, period)),
                err_msg=f"{variable} at {period}",
            )


def _assert_same_population(simulation, expected):
    for key, population in expected.populations.items():
        other = simulation.populations[key]
        assert other.count == population.count
        np.testing.assert_array_equal(other.ids, population.ids)
        if not population.entity.is_person:
            np.testing.assert_array_equal(
                other.members_entity_id, population.members_entity_id
            )
            np.testing.assert_array_equal(other.members_role, population.members_role)


PARTIAL_SALARY = {
    "persons": {"p0": {"salary": {PERIOD: 3000.0}}, "p1": {}},
    "households": {"h0": {"parents": ["p0", "p1"]}},
}


@pytest.mark.filterwarnings("ignore:You cannot set a value")
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    spec=reform_specs(),
    situation=situations() | situations_without_entities(),
)
# The reproduction in #612: a salary given to one person of two.
@example(spec=((("salary", 1000.0),), (), False, None), situation=PARTIAL_SALARY)
# The same along an axis over the other person.
@example(
    spec=((("salary", 1000.0),), (), False, None),
    situation={
        **PARTIAL_SALARY,
        "axes": [
            [
                {
                    "name": "salary",
                    "count": 2,
                    "index": 1,
                    "min": 0,
                    "max": 5000,
                    "period": PERIOD,
                }
            ]
        ],
    },
)
# A household's enum given to one household of two, a yearly rent to the other.
@example(
    spec=((("housing_occupancy_status", "owner"), ("rent", 500.0)), (), False, None),
    situation={
        "persons": {"p0": {}, "p1": {}},
        "households": {
            "h0": {"parents": ["p0"], "housing_occupancy_status": {PERIOD: "homeless"}},
            "h1": {"parents": ["p1"], "rent": {"2022": 6000.0}},
        },
    },
)
# Without entities, the builder sets each variable through ``set_input``.
@example(
    spec=((("salary", 1000.0),), ("income_tax",), False, None),
    situation={"salary": {PERIOD: [3000.0, 0.0]}, "income_tax": {PERIOD: [7.0, 9.0]}},
)
# An input the reform neutralizes, and one only the reform has.
@example(
    spec=((), ("salary",), True, None),
    situation={
        "persons": {
            "p0": {"salary": {PERIOD: 3000.0}, ADDED: {PERIOD: 9.0}},
            "p1": {"salary": {"2022": 1200.0}},
        },
        "households": {"h0": {"parents": ["p0", "p1"]}},
    },
)
def test_the_baseline_is_what_the_builder_makes_of_the_situation(spec, situation):
    added = spec[2]
    if not added:
        # Only a reform that adds the variable takes it.
        situation = _without_added(situation)
    reform = _reform(spec)
    simulation = Simulation(situation=copy.deepcopy(situation), reform=reform)
    baseline = simulation.baseline
    expected = _built(_without_added(situation), SYSTEM)

    _assert_calculates_the_same(baseline, expected)
    _assert_same_population(baseline, expected)
    assert baseline._user_input_keys == expected._user_input_keys
    assert sorted(baseline.input_variables) == sorted(expected.input_variables)
    assert baseline.to_input_dict() == expected.to_input_dict()

    assert simulation.branches["baseline"] is baseline
    assert baseline.parent_branch is simulation
    assert baseline.tax_benefit_system is SYSTEM

    reform_system = Simulation.default_tax_benefit_system(reform=reform)
    _assert_calculates_the_same(simulation, _built(situation, reform_system))
