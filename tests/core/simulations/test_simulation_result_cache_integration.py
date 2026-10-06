from __future__ import annotations

import numpy as np

from policyengine_core.country_template import situation_examples
from policyengine_core.periods import period
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.simulations.simulation_result_cache import ResultCacheKey


def _simulation(tax_benefit_system):
    return SimulationBuilder().build_from_entities(
        tax_benefit_system,
        situation_examples.single,
    )


def test_calculation_uses_the_owned_result_cache(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    value = simulation.calculate("income_tax", "2017-01")

    assert (
        simulation.result_cache.get(ResultCacheKey("income_tax", period("2017-01")))
        is value
    )


def test_delete_arrays_removes_matching_input_provenance(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))

    simulation.delete_arrays("salary", "2017-01")

    assert simulation.supplied_input_periods("salary") == []


def test_delete_larger_period_removes_subperiod_input_provenance(
    tax_benefit_system,
) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    simulation.set_input("salary", "2017-02", np.array([5_000]))

    simulation.delete_arrays("salary", "2017")

    assert simulation.supplied_input_periods("salary") == []


def test_clear_calculated_results_keeps_supplied_values(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    simulation.calculate("income_tax", "2017-01")

    simulation.clear_calculated_results()

    assert np.array_equal(
        simulation.get_supplied_input("salary", "2017-01"),
        np.array([4_000], dtype=np.float32),
    )
    assert ResultCacheKey("income_tax", period("2017-01")) not in (
        simulation.result_cache
    )


def test_supplied_input_queries_follow_branch_ancestry(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    branch = simulation.get_branch("reform")
    branch.set_input("salary", "2017-02", np.array([5_000]))

    assert branch.supplied_input_periods("salary") == [
        period("2017-01"),
        period("2017-02"),
    ]
    assert simulation.supplied_input_periods("salary") == [period("2017-01")]


def test_get_supplied_input_prefers_the_current_branch(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    branch = simulation.get_branch("reform")
    branch.set_input("salary", "2017-01", np.array([5_000]))

    assert np.array_equal(
        branch.get_supplied_input("salary", "2017-01"),
        np.array([5_000], dtype=np.float32),
    )


def test_retain_supplied_inputs_filters_variables(tax_benefit_system) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    simulation.set_input("age", "2017-01", np.array([40]))

    simulation.retain_supplied_inputs(["salary"])

    assert simulation.get_supplied_input("salary", "2017-01") is not None
    assert simulation.get_supplied_input("age", "2017-01") is None


def test_clone_has_independent_result_and_provenance_indexes(
    tax_benefit_system,
) -> None:
    simulation = _simulation(tax_benefit_system)
    simulation.set_input("salary", "2017-01", np.array([4_000]))
    simulation.calculate("income_tax", "2017-01")

    clone = simulation.clone()
    clone.delete_arrays("salary", "2017-01")

    assert len(clone.result_cache) == 0
    assert simulation.get_supplied_input("salary", "2017-01") is not None
    assert clone.get_supplied_input("salary", "2017-01") is None
