from policyengine_core.country_template.situation_examples import single
from policyengine_core.country_template import Simulation as CountryTemplateSimulation
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
from policyengine_core.model_api import Variable
from policyengine_core.periods import MONTH
from policyengine_core.simulations import SimulationBuilder
import policyengine_core.simulations.simulation as simulation_module
from policyengine_core.simulations.simulation_macro_cache import (
    SimulationMacroCache,
)
import importlib.metadata
import numpy as np
import pandas as pd
from pathlib import Path


def test_calculate_full_tracer(tax_benefit_system):
    simulation = SimulationBuilder().build_default_simulation(tax_benefit_system)
    simulation.trace = True
    simulation.calculate("income_tax", "2017-01")

    income_tax_node = simulation.tracer.trees[0]
    assert income_tax_node.name == "income_tax"
    assert str(income_tax_node.period) == "2017-01"
    assert income_tax_node.value == 0

    salary_node = income_tax_node.children[0]
    assert salary_node.name == "salary"
    assert str(salary_node.period) == "2017-01"


def test_get_entity_not_found(tax_benefit_system):
    simulation = SimulationBuilder().build_default_simulation(tax_benefit_system)
    assert simulation.get_entity(plural="no_such_entities") is None


def test_clone(tax_benefit_system):
    simulation = SimulationBuilder().build_from_entities(
        tax_benefit_system,
        {
            "persons": {
                "bill": {"salary": {"2017-01": 3000}},
            },
            "households": {"household": {"parents": ["bill"]}},
        },
    )

    simulation_clone = simulation.clone()
    assert simulation != simulation_clone

    for entity_id, entity in simulation.populations.items():
        assert entity != simulation_clone.populations[entity_id]

    assert simulation.persons != simulation_clone.persons

    salary_holder = simulation.person.get_holder("salary")
    salary_holder_clone = simulation_clone.person.get_holder("salary")

    assert salary_holder != salary_holder_clone
    assert salary_holder_clone.simulation == simulation_clone
    assert salary_holder_clone.population == simulation_clone.persons


def test_get_memory_usage(tax_benefit_system):
    simulation = SimulationBuilder().build_from_entities(tax_benefit_system, single)
    simulation.calculate("disposable_income", "2017-01")
    memory_usage = simulation.get_memory_usage(variables=["salary"])
    assert memory_usage["total_nb_bytes"] > 0
    assert len(memory_usage["by_variable"]) == 1


def test_macro_cache(tax_benefit_system):
    simulation = SimulationBuilder().build_from_entities(tax_benefit_system, single)

    cache = SimulationMacroCache(tax_benefit_system)
    assert cache.core_version == importlib.metadata.version("policyengine-core")
    assert cache.country_version == "0.0.0"

    cache.set_cache_path(
        parent_path="tests/core",
        dataset_name="test_dataset",
        variable_name="test_variable",
        period="2020",
        branch_name="test_branch",
    )
    cache.set_cache_value(
        cache_file_path=cache.cache_file_path,
        value=np.array([1, 2, 3], dtype=np.float32),
    )
    assert cache.get_cache_path() == Path(
        "tests/core/test_dataset_variable_cache/test_variable_2020_test_branch.h5"
    )
    assert np.array_equal(
        cache.get_cache_value(cache.cache_file_path),
        np.array([1, 2, 3], dtype=np.float32),
    )
    cache.clear_cache(cache.cache_folder_path)
    assert not cache.cache_folder_path.exists()


def test_calculate_without_macro_cache_does_not_build_macro_cache(
    tax_benefit_system, monkeypatch
):
    class UnexpectedSimulationMacroCache:
        def __init__(self, tax_benefit_system):
            raise AssertionError(
                "SimulationMacroCache should not be constructed when "
                "macro cache reads are disabled"
            )

    monkeypatch.setattr(
        simulation_module,
        "SimulationMacroCache",
        UnexpectedSimulationMacroCache,
    )

    simulation = SimulationBuilder().build_default_simulation(tax_benefit_system)

    simulation.calculate("income_tax", "2017-01")


class formula_component_for_safe_export(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Formula component for safe export tests."

    def formula(person, period):
        return person("salary", period) * 0


class pseudo_input_for_safe_export(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Pseudo-input for safe export tests."
    adds = ["formula_component_for_safe_export"]


def _safe_export_dataset(dataframe):
    return Dataset.from_dataframe(dataframe, "2022-01")


def _safe_export_simulation(isolated_tax_benefit_system):
    isolated_tax_benefit_system.add_variable(formula_component_for_safe_export)
    isolated_tax_benefit_system.add_variable(pseudo_input_for_safe_export)

    dataframe = pd.DataFrame(
        {
            "person_id__2022": [0],
            "household_id__2022": [0],
            "person_household_id__2022": [0],
            "person_household_role__2022": ["parent"],
            "household_weight__2022": [1.0],
            "salary__2022-01": [0.0],
            "pseudo_input_for_safe_export__2022-01": [999.0],
        }
    )
    return CountryTemplateSimulation(
        tax_benefit_system=isolated_tax_benefit_system,
        dataset=_safe_export_dataset(dataframe),
    )


def test__given_pseudo_input_in_dataset__then_input_dataframe_excludes_it(
    isolated_tax_benefit_system,
):
    # Given
    simulation = _safe_export_simulation(isolated_tax_benefit_system)

    assert simulation.calculate("pseudo_input_for_safe_export", "2022-01")[0] == 999.0

    # When
    dataframe = simulation.to_input_dataframe()
    reloaded = CountryTemplateSimulation(
        tax_benefit_system=isolated_tax_benefit_system,
        dataset=_safe_export_dataset(dataframe),
    )

    # Then
    assert "salary__2022-01" in dataframe.columns
    assert "pseudo_input_for_safe_export__2022-01" not in dataframe.columns
    assert "salary" in simulation.true_input_variables
    assert "pseudo_input_for_safe_export" not in simulation.true_input_variables
    assert (
        "pseudo_input_for_safe_export__2022-01"
        in simulation.to_input_dataframe(include_computed_variables=True).columns
    )
    assert reloaded.calculate("pseudo_input_for_safe_export", "2022-01")[0] == 0.0


def test__given_pseudo_input_in_dataset__then_input_dict_h5_round_trip_excludes_it(
    isolated_tax_benefit_system, tmp_path
):
    # Given
    simulation = _safe_export_simulation(isolated_tax_benefit_system)
    exported_data = simulation.to_input_dict()
    h5_path = tmp_path / "safe_export.h5"

    class SafeExportDataset(Dataset):
        name = "safe_export"
        label = "Safe export"
        file_path = h5_path
        data_format = Dataset.TIME_PERIOD_ARRAYS

    # When
    SafeExportDataset().save_dataset(exported_data)
    reloaded = CountryTemplateSimulation(
        tax_benefit_system=isolated_tax_benefit_system,
        dataset=Dataset.from_file(h5_path),
    )

    # Then
    assert "salary" in exported_data
    assert "pseudo_input_for_safe_export" not in exported_data
    assert "pseudo_input_for_safe_export" in simulation.to_input_dict(
        include_computed_variables=True
    )
    assert reloaded.calculate("pseudo_input_for_safe_export", "2022-01")[0] == 0.0
