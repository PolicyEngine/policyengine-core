"""Writing a situation into a simulation is the same as building it anew.

For any dataset, any reform (or none), any calculations, branches and
earlier situations before it, and any situation (with entities or without,
with axes or without), once ``SimulationBuilder.build_from_dict`` writes the
situation into the simulation:

* the simulation is not over a dataset, and has none;
* it calculates what a simulation the builder makes from the situation
  alone, under the same system, calculates (differential);
* it records the same inputs, input variables and axes as
  ``Simulation(situation=..., reform=...)``, and exports the same inputs;
  its baseline calculates and records what that simulation's baseline does;
* of its branches, only the baseline is left.

The baseline is compared with ``Simulation(situation=..., reform=...)``'s,
not with a simulation built under the baseline system: given a variable for
some people only (or along axes), the situation builder stores the reform's
default for the others as an input, in both (see
``test_reform_baseline_system.py``).

Examples, including the ones that failed before, are in
``test_builder_replaces_population.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.country_template import Simulation  # noqa: E402
from policyengine_core.simulations import SimulationBuilder  # noqa: E402
from tests.core.test_reform_baseline_system import (  # noqa: E402
    add_doubled_salary,
    change_salary_default,
    neutralize_income_tax,
)

PERIOD = "2022-01"
REFORMS = {
    None: None,
    "rate": {"taxes.income_tax_rate": {"2022-01-01": 0.42}},
    "neutralize income tax": neutralize_income_tax,
    "salary default": change_salary_default,
    "added variable": add_doubled_salary,
}
OUTPUTS = {
    "salary": PERIOD,
    "income_tax": PERIOD,
    "disposable_income": PERIOD,
    "household_income": PERIOD,
    "age": PERIOD,
    "household_weight": "2022",
    "doubled_salary": PERIOD,
}
AMOUNT = st.integers(min_value=0, max_value=10_000).map(float)


@st.composite
def data(draw):
    """People in households of one or two, with salaries and weights."""
    sizes = draw(st.lists(st.integers(1, 2), min_size=1, max_size=4))
    household = [index for index, size in enumerate(sizes, 1) for _ in range(size)]
    count = len(household)
    columns = {
        "person_id__2022": list(range(1, count + 1)),
        "household_id__2022": household,
        "person_household_id__2022": household,
        "household_weight__2022": [100.0 * h for h in household],
        "salary__2022-01": draw(st.lists(AMOUNT, min_size=count, max_size=count)),
    }
    if draw(st.booleans()):
        columns["age__2022-01"] = [30 + h for h in household]
    return pd.DataFrame(columns)


@st.composite
def situations(draw):
    """One household of up to three people, or people without entities."""
    count = draw(st.integers(1, 3))
    salaries = draw(
        st.lists(st.one_of(st.none(), AMOUNT), min_size=count, max_size=count)
    )
    if draw(st.booleans()):
        # No entities: every person given a salary.
        return {"salary": {PERIOD: [amount or 0.0 for amount in salaries]}}
    people = [f"person_{index}" for index in range(count)]
    situation = {
        "persons": {
            person: {} if amount is None else {"salary": {PERIOD: amount}}
            for person, amount in zip(people, salaries)
        },
        "households": {"home": {"parents": people[:2], "children": people[2:]}},
    }
    if draw(st.booleans()):
        situation["axes"] = [
            [
                {
                    "name": "salary",
                    "count": draw(st.integers(2, 3)),
                    "min": 0,
                    "max": 5000,
                    "period": PERIOD,
                }
            ]
        ]
    return situation


STEPS = st.lists(
    st.one_of(
        st.tuples(st.just("calculate"), st.sampled_from(list(OUTPUTS))),
        st.tuples(st.just("calculate on the baseline"), st.sampled_from(list(OUTPUTS))),
        st.tuples(st.just("branch"), st.sampled_from(list(OUTPUTS))),
        st.tuples(st.just("write a situation"), situations()),
    ),
    max_size=4,
)


def _write(simulation, situation):
    builder = SimulationBuilder()
    builder.default_period = simulation.default_input_period
    return builder.build_from_dict(
        simulation.tax_benefit_system, situation, simulation=simulation
    )


def _calculate(simulation, variable):
    if variable in simulation.tax_benefit_system.variables:
        simulation.calculate(variable, OUTPUTS[variable])


def _assert_calculates_as(simulation, reference, label):
    for variable, period in OUTPUTS.items():
        if variable not in reference.tax_benefit_system.variables:
            continue
        np.testing.assert_array_equal(
            simulation.calculate(variable, period),
            reference.calculate(variable, period),
            err_msg=f"{label} {variable}",
        )


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    data_=data(),
    reform=st.sampled_from(list(REFORMS)),
    steps=STEPS,
    situation=situations(),
)
# Values calculated for the data, on the simulation and its baseline.
@example(
    data_=pd.DataFrame(
        {
            "person_id__2022": [1, 2],
            "household_id__2022": [1, 2],
            "person_household_id__2022": [1, 2],
            "household_weight__2022": [100.0, 200.0],
            "salary__2022-01": [1000.0, 2000.0],
        }
    ),
    reform="rate",
    steps=[("calculate", "income_tax"), ("calculate on the baseline", "income_tax")],
    situation={
        "persons": {"person_0": {"salary": {PERIOD: 3000.0}}},
        "households": {"home": {"parents": ["person_0"]}},
    },
)
def test_writing_a_situation_is_building_it_anew(data_, reform, steps, situation):
    simulation = Simulation(dataset=data_, reform=REFORMS[reform])
    for index, (kind, argument) in enumerate(steps):
        if kind == "calculate":
            _calculate(simulation, argument)
        elif kind == "calculate on the baseline" and simulation.baseline is not None:
            _calculate(simulation.baseline, argument)
        elif kind == "branch":
            _calculate(simulation.get_branch(f"branch_{index}"), argument)
        elif kind == "write a situation":
            _write(simulation, argument)
            _calculate(simulation, "income_tax")

    _write(simulation, situation)

    assert not simulation.is_over_dataset
    assert simulation.dataset is None
    assert set(simulation.branches) == ({"baseline"} if reform else set())

    # Built by the builder alone, under the same system.
    builder = SimulationBuilder()
    builder.default_period = PERIOD
    _assert_calculates_as(
        simulation,
        builder.build_from_dict(simulation.tax_benefit_system, situation),
        "simulation",
    )

    # Built from the situation by ``Simulation``.
    reference = Simulation(situation=situation, reform=REFORMS[reform])
    assert simulation._user_input_keys == reference._user_input_keys
    assert simulation.input_variables == reference.input_variables
    assert simulation.has_axes == reference.has_axes
    pd.testing.assert_frame_equal(
        simulation.to_input_dataframe(), reference.to_input_dataframe()
    )
    if reform:
        assert simulation.baseline is simulation.branches["baseline"]
        assert simulation.baseline.baseline is None
        assert not simulation.baseline.is_over_dataset
        _assert_calculates_as(simulation.baseline, reference.baseline, "baseline")
        assert simulation.baseline._user_input_keys == (
            reference.baseline._user_input_keys
        )
        assert simulation.baseline.input_variables == reference.baseline.input_variables
    else:
        assert simulation.baseline is None
