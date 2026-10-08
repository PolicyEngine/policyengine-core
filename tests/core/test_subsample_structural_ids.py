"""Subsampling uses the loaded population when structural IDs have formulas.

The source partition and roles are leaves: a structural formula's cache must
neither replace them nor decide which people constitute a whole household.
"""

import numpy as np
import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem, Simulation
from policyengine_core.country_template.entities import Household, Person
from policyengine_core.data import Dataset
from policyengine_core.periods import ETERNITY
from policyengine_core.simulations.simulation_builder import SimulationBuilder
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import (
    DATASET_YEAR,
    HOUSEHOLD_OF_PERSON,
    base,
    dataframe,
    doubled,
    household_total,
    stored,
)


class household_id(Variable):
    value_type = int
    entity = Household
    definition_period = ETERNITY
    label = "Formula household identifier"

    def formula(household, period):
        # Deliberately unsuitable for deciding population membership.
        return household.filled_array(-999)


class person_id(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Formula person identifier"

    def formula(person, period):
        return np.arange(person.count) + 700


class person_household_id(Variable):
    value_type = int
    entity = Person
    definition_period = ETERNITY
    label = "Formula person household identifier"

    def formula(person, period):
        return person.filled_array(-999)


class person_household_role(Variable):
    value_type = str
    entity = Person
    definition_period = ETERNITY
    label = "Formula person household role"

    def formula(person, period):
        return person.filled_array("child")


STRUCTURAL_VARIABLES = (
    household_id,
    person_id,
    person_household_id,
    person_household_role,
)


def partition(simulation):
    return {
        frozenset(
            simulation.persons.ids[
                simulation.household.members_entity_id == index
            ].tolist()
        )
        for index in range(simulation.household.count)
    }


def roles_by_person(simulation):
    return {
        person: getattr(role, "key", role)
        for person, role in zip(
            simulation.persons.ids, simulation.household.members_role
        )
    }


def make_population_simulation(recorded_structural_inputs, shuffled):
    """A population with structural leaves and optional formula overrides."""
    system = CountryTaxBenefitSystem()
    system.add_variables(base, doubled, household_total)
    for variable in STRUCTURAL_VARIABLES:
        system.replace_variable(variable)

    data = dataframe()
    labels = {1: 901, 2: -20, 3: 33, 4: 88, 5: 205, 6: 411}
    membership = np.array([labels[group] for group in HOUSEHOLD_OF_PERSON])
    # Explicit first/second parent and child roles make role loss observable.
    roles = np.array([0, 2, 0, 1, 2, 0, 0, 2, 0, 0])
    order = np.array([9, 0, 4, 6, 2, 8, 1, 5, 3, 7]) if shuffled else np.arange(10)
    data = data.iloc[order].reset_index(drop=True)
    membership, roles = membership[order], roles[order]

    builder = SimulationBuilder()
    builder.create_entities(system)
    builder.declare_person_entity("person", data[f"person_id__{DATASET_YEAR}"])
    household = builder.declare_entity("household", list(labels.values()))
    builder.join_with_persons(household, membership, roles)
    simulation = Simulation(
        tax_benefit_system=system,
        populations=builder.populations,
        default_calculation_period=DATASET_YEAR,
    )
    # Attach period metadata without reloading the explicitly built population.
    simulation.dataset = Dataset.from_dataframe(data, DATASET_YEAR)
    simulation.is_over_dataset = True
    simulation.set_input("base", DATASET_YEAR, data[f"base__{DATASET_YEAR}"].values)
    simulation.set_input(
        "household_weight", DATASET_YEAR, [100, 200, 300, 400, 500, 600]
    )
    if recorded_structural_inputs:
        simulation.set_input("person_id", ETERNITY, simulation.persons.ids)
        simulation.set_input("person_household_id", ETERNITY, membership)
        simulation.set_input(
            "person_household_role",
            ETERNITY,
            [role.key for role in simulation.household.members_role],
        )
    return simulation


def test_formula_only_household_id_uses_recorded_membership():
    """The reviewed failure: no household ID source gives an IndexError."""
    data = dataframe().drop(columns=[f"household_id__{DATASET_YEAR}"])
    system = CountryTaxBenefitSystem()
    system.replace_variable(household_id)
    system.add_variables(base, doubled, household_total)
    simulation = Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(data, DATASET_YEAR),
    )
    source_partition = partition(simulation)
    simulation.calculate("household_id", ETERNITY)

    simulation.subsample(n=4, seed="formula-household-id", time_period=DATASET_YEAR)

    assert partition(simulation) <= source_partition
    assert 1 <= simulation.household.count <= 4
    assert not any(name == "household_id" for name, _, _ in stored(simulation))


@pytest.mark.parametrize("recorded_structural_inputs", [False, True])
@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("quantize_weights", [False, True])
@pytest.mark.parametrize("n,seed", [(1, "one"), (4, "four"), (8, "repeats")])
def test_loaded_structural_leaves_survive_independent_of_formula_caches(
    recorded_structural_inputs, shuffled, quantize_weights, n, seed
):
    """Differential: formula caches cannot alter IDs, groups, roles or sums."""
    fresh = make_population_simulation(recorded_structural_inputs, shuffled)
    used = make_population_simulation(recorded_structural_inputs, shuffled)
    source_partition = partition(fresh)
    source_roles = roles_by_person(fresh)
    source_totals = dict(
        zip(
            fresh.persons.ids,
            fresh.calculate("household_total", DATASET_YEAR, map_to="person"),
        )
    )
    for variable in STRUCTURAL_VARIABLES:
        used.calculate(variable.__name__, ETERNITY)

    for simulation in (fresh, used):
        simulation.subsample(
            n=n,
            seed=seed,
            time_period=DATASET_YEAR,
            quantize_weights=quantize_weights,
        )
        assert partition(simulation) <= source_partition
        assert 1 <= simulation.household.count <= min(n, 6)
        assert roles_by_person(simulation) == {
            person: source_roles[person] for person in simulation.persons.ids
        }
        np.testing.assert_array_equal(
            simulation.calculate("household_total", DATASET_YEAR, map_to="person"),
            [source_totals[person] for person in simulation.persons.ids],
        )
        assert not any(name == "household_id" for name, _, _ in stored(simulation))

    assert partition(used) == partition(fresh)
    assert stored(used) == stored(fresh)
