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
from policyengine_core.country_template.entities import Person
from policyengine_core.errors import SituationParsingError, VariableNotFoundError
from policyengine_core.periods import MONTH
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.variables import Variable

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


@pytest.mark.parametrize(
    "situation, error",
    [
        # A value without a period, and no default period to give it.
        ({"salary": SALARY}, SituationParsingError),
        ({"wages": {PERIOD: [SALARY]}}, VariableNotFoundError),
    ],
)
def test_variables_that_fail_to_parse_leave_the_data(
    tax_benefit_system, situation, error
):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())

    with pytest.raises(error):
        SimulationBuilder().build_from_dict(
            tax_benefit_system, situation, simulation=simulation
        )

    # Before, the data's populations were resized to the situation's first.
    assert simulation.persons.count == 10
    assert simulation.is_over_dataset
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1000.0] * 10)


def test_a_situation_that_fails_while_written_leaves_what_was_written(
    tax_benefit_system,
):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
    simulation.get_branch("scenario")

    with pytest.raises(ValueError):
        SimulationBuilder().build_from_dict(
            tax_benefit_system,
            {"salary": {PERIOD: [SALARY]}, "age": {PERIOD: ["forty"]}},
            simulation=simulation,
        )

    # Replaced, with the salary written before the age failed, and records
    # and branches that describe that.
    assert not simulation.is_over_dataset
    assert simulation.dataset is None
    assert simulation.branches == {}
    assert simulation.input_variables == ["salary"]
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [SALARY])


def test_a_branch_cannot_be_overwritten(tax_benefit_system):
    simulation = Simulation(tax_benefit_system=tax_benefit_system, dataset=_data())
    branch = simulation.get_branch("scenario")

    # It shares its parent's population and baseline.
    with pytest.raises(ValueError, match="branch 'scenario'"):
        SimulationBuilder().build_from_dict(
            tax_benefit_system, SITUATION, simulation=branch
        )

    assert branch.persons.count == 10
    assert branch.is_over_dataset


class overwrites_its_simulation(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Salary, read before overwriting the simulation calculating it"

    def formula(person, period, parameters):
        salary = person("salary", period)
        simulation = person.simulation
        SimulationBuilder().build_default_simulation(
            simulation.tax_benefit_system, 1, simulation
        )
        return salary


def test_a_simulation_cannot_be_overwritten_while_it_calculates(
    isolated_tax_benefit_system,
):
    isolated_tax_benefit_system.add_variable(overwrites_its_simulation)
    simulation = Simulation(
        tax_benefit_system=isolated_tax_benefit_system, dataset=_data()
    )

    # Before, it returned the ten people's salaries for a one-person
    # simulation, which failed when stored.
    with pytest.raises(RuntimeError, match="while it is calculating"):
        simulation.calculate("overwrites_its_simulation", PERIOD)

    assert simulation.persons.count == 10
    assert simulation.is_over_dataset


class taxed_salary(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Salary, once income tax has been requested"
    requires_computation_after = "income_tax"

    def formula(person, period, parameters):
        return person("salary", period)


def test_a_prerequisite_requested_for_the_data_is_not_requested_for_the_situation(
    isolated_tax_benefit_system,
):
    isolated_tax_benefit_system.add_variable(taxed_salary)
    simulation = Simulation(
        tax_benefit_system=isolated_tax_benefit_system, dataset=_data()
    )
    simulation.calculate("income_tax", PERIOD)
    simulation.calculate("taxed_salary", PERIOD)

    SimulationBuilder().build_from_dict(
        isolated_tax_benefit_system, SITUATION, simulation=simulation
    )

    # As in a simulation built from the situation alone.
    reference = SimulationBuilder().build_from_dict(
        isolated_tax_benefit_system, SITUATION
    )
    for built in (reference, simulation):
        with pytest.raises(ValueError, match="requires income_tax to be requested"):
            built.calculate("taxed_salary", PERIOD)


def test_a_trace_shows_the_situations_values(tax_benefit_system):
    simulation = Simulation(
        tax_benefit_system=tax_benefit_system, dataset=_data(), trace=True
    )
    simulation.calculate("income_tax", PERIOD)

    SimulationBuilder().build_from_dict(
        tax_benefit_system, SITUATION, simulation=simulation
    )
    simulation.calculate("income_tax", PERIOD)

    # Before, the trace kept the first calculation's (the data's) values.
    trace = simulation.tracer.get_serialized_flat_trace()
    np.testing.assert_allclose(
        trace[f"income_tax<{PERIOD}, (default)>"]["value"], [450]
    )


def test_a_situation_written_over_another_replaces_it():
    first = {
        **SITUATION,
        "axes": [[{"name": "salary", "count": 3, "min": 0, "max": 6000}]],
    }
    simulation = CountryTemplateSimulation(situation=first)
    simulation.calculate("income_tax", PERIOD)
    assert simulation.has_axes

    builder = SimulationBuilder()
    builder.default_period = PERIOD
    builder.build_from_dict(
        simulation.tax_benefit_system, SITUATION, simulation=simulation
    )

    # The first situation's axes and its record as the situation the
    # simulation was built from describe it no longer.
    assert not simulation.has_axes
    assert simulation.situation_input is None
    _assert_same_values(simulation, CountryTemplateSimulation(situation=SITUATION))


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
