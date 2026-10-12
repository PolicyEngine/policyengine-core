"""``SimulationBuilder`` replaces the population of a simulation it is given.

``build_from_dict`` (like ``build_from_entities``, ``build_from_variables``
and ``build_default_simulation``) writes a situation into a simulation passed
as ``simulation``. It wrote the situation into that simulation's populations
as they were, so a simulation built over a dataset and then given a
situation:

* kept ``is_over_dataset`` and ``dataset``, so the macro cache, and country
  models that read ``is_over_dataset``, treated the situation as data;
* answered from its fast cache with values calculated for the dataset;
* kept the dataset's arrays for every variable the situation did not set;
* kept the dataset's input record, so ``to_input_dataframe`` failed;
* kept its branches, the baseline among them, on the dataset.

The property-based version, which compares against a simulation built from
the situation alone, is in ``test_builder_replaces_population_property.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_core.country_template import Microsimulation
from policyengine_core.country_template import Simulation as CountryTemplateSimulation
from policyengine_core.errors import SituationParsingError
from policyengine_core.simulations import Simulation, SimulationBuilder

PERIOD = "2022-01"
SALARY = 3000.0
SITUATION = {
    "persons": {"you": {"salary": {PERIOD: SALARY}}},
    "households": {"home": {"parents": ["you"]}},
}
REFORM = {"taxes.income_tax_rate": {"2022-01-01": 0.4}}
# Variables the situation sets (salary) or leaves to their defaults and
# formulas, including ones the data sets.
OUTPUTS = ["salary", "income_tax", "disposable_income", "age", "household_weight"]


def _data() -> pd.DataFrame:
    """Ten people in five households, each with an age and salary."""
    return pd.DataFrame(
        {
            "person_id__2022": list(range(1, 11)),
            "household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "person_household_id__2022": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "household_weight__2022": [100.0] * 10,
            "salary__2022-01": [1000.0] * 10,
            "age__2022-01": [40] * 10,
        }
    )


def _period(variable: str) -> str:
    return "2022" if variable == "household_weight" else PERIOD


def _assert_same_values(simulation, reference, label=""):
    for variable in OUTPUTS:
        np.testing.assert_array_equal(
            simulation.calculate(variable, _period(variable)),
            reference.calculate(variable, _period(variable)),
            err_msg=f"{label} {variable}",
        )


def test_a_situation_written_over_data_is_not_over_a_dataset(tax_benefit_system):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
    assert simulation.is_over_dataset
    assert simulation.dataset is not None

    SimulationBuilder().build_from_dict(
        tax_benefit_system, SITUATION, simulation=simulation
    )

    assert not simulation.is_over_dataset
    assert simulation.dataset is None


def test_it_holds_nothing_from_the_data(tax_benefit_system):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
    # Calculated for the data, these filled the fast cache and holders.
    for variable in OUTPUTS:
        simulation.calculate(variable, _period(variable))

    SimulationBuilder().build_from_dict(
        tax_benefit_system, SITUATION, simulation=simulation
    )

    reference = SimulationBuilder().build_from_dict(tax_benefit_system, SITUATION)
    # Before, income tax was the ten people's (150 each) and age and weight
    # the data's.
    np.testing.assert_allclose(simulation.calculate("income_tax", PERIOD), [450])
    _assert_same_values(simulation, reference)
    assert simulation.input_variables == reference.input_variables


def test_its_inputs_are_the_situations(tax_benefit_system):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())

    SimulationBuilder().build_from_dict(
        tax_benefit_system, SITUATION, simulation=simulation
    )

    reference = SimulationBuilder().build_from_dict(tax_benefit_system, SITUATION)
    assert simulation.true_input_variables == ["salary"]
    # Before, the export failed: the data's ten-person inputs were still
    # recorded on a one-person population.
    pd.testing.assert_frame_equal(
        simulation.to_input_dataframe(), reference.to_input_dataframe()
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda builder, system, simulation: builder.build_from_entities(
                system, SITUATION, simulation
            ),
            id="build_from_entities",
        ),
        pytest.param(
            lambda builder, system, simulation: builder.build_from_variables(
                system, {"salary": {PERIOD: [SALARY, 0.0]}}, simulation
            ),
            id="build_from_variables",
        ),
        pytest.param(
            lambda builder, system, simulation: builder.build_default_simulation(
                system, 2, simulation
            ),
            id="build_default_simulation",
        ),
        pytest.param(
            lambda builder, system, simulation: builder.build_from_dict(
                system, {"salary": {PERIOD: [SALARY, 0.0]}}, simulation
            ),
            id="build_from_dict without entities",
        ),
    ],
)
def test_every_builder_replaces_the_population(tax_benefit_system, build):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
    simulation.calculate("income_tax", PERIOD)
    simulation.get_branch("scenario")

    build(SimulationBuilder(), tax_benefit_system, simulation)

    reference = build(SimulationBuilder(), tax_benefit_system, None)
    assert not simulation.is_over_dataset
    assert simulation.dataset is None
    # A branch of the data's population is forgotten.
    assert simulation.branches == {}
    _assert_same_values(simulation, reference)


def test_a_situation_that_fails_to_parse_leaves_the_data(tax_benefit_system):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())

    with pytest.raises(SituationParsingError):
        SimulationBuilder().build_from_dict(
            tax_benefit_system,
            {**SITUATION, "companies": {"acme": {}}},
            simulation=simulation,
        )

    assert simulation.is_over_dataset
    assert simulation.dataset is not None
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1000.0] * 10)


def test_a_reform_simulation_compares_against_the_situation():
    simulation = CountryTemplateSimulation(dataset=_data(), reform=REFORM)
    simulation.calculate("income_tax", PERIOD)
    simulation.baseline.calculate("income_tax", PERIOD)

    SimulationBuilder().build_from_dict(
        simulation.tax_benefit_system, SITUATION, simulation=simulation
    )

    # The baseline branch is rebuilt on the situation, under the baseline
    # policy, and is the simulation's baseline.
    assert simulation.baseline is simulation.branches["baseline"]
    assert list(simulation.branches) == ["baseline"]
    assert simulation.baseline.baseline is None
    assert not simulation.baseline.is_over_dataset
    _assert_same_values(
        simulation, CountryTemplateSimulation(situation=SITUATION, reform=REFORM)
    )
    _assert_same_values(
        simulation.baseline, CountryTemplateSimulation(situation=SITUATION)
    )
    np.testing.assert_allclose(
        simulation.calculate("income_tax", PERIOD), [SALARY * 0.4]
    )
    np.testing.assert_allclose(
        simulation.baseline.calculate("income_tax", PERIOD), [SALARY * 0.15]
    )


def test_the_macro_cache_is_not_read_for_a_situation(isolated_tax_benefit_system):
    # The country template's dataset is stored as time-period arrays, which
    # the macro cache can serve, and income tax depends on one parameter.
    isolated_tax_benefit_system.variables[
        "income_tax"
    ].exhaustive_parameter_dependencies = ["taxes.income_tax_rate"]
    simulation = Microsimulation(tax_benefit_system=isolated_tax_benefit_system)
    simulation.macro_cache_read = True
    assert simulation.check_macro_cache("income_tax", PERIOD)

    SimulationBuilder().build_from_dict(
        isolated_tax_benefit_system, SITUATION, simulation=simulation
    )

    assert not simulation.check_macro_cache("income_tax", PERIOD)


@pytest.mark.parametrize("data_first", [True, False])
def test_only_a_simulation_with_a_dataset_can_be_subsampled(
    tax_benefit_system, data_first
):
    simulation = (
        Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
        if data_first
        else None
    )
    simulation = SimulationBuilder().build_from_dict(
        tax_benefit_system, SITUATION, simulation=simulation
    )

    # Before, one written over data subsampled the data's leftover inputs and
    # failed on their length, and one built from a situation alone raised an
    # AttributeError reading the default seed (the dataset's name).
    with pytest.raises(ValueError, match="Only a simulation with a dataset"):
        simulation.subsample(n=1)
