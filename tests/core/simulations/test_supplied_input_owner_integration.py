"""Type D integration: public input writes, policy reuse, and branch ownership."""

from unittest.mock import patch

import numpy as np
import pytest

from policyengine_core.country_template import situation_examples
from policyengine_core.experimental import MemoryConfig
from policyengine_core.model_api import Variable
from policyengine_core.periods import period
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.simulations.simulation_result_cache import ResultCacheKey


@pytest.fixture
def simulation(tax_benefit_system):
    return SimulationBuilder().build_from_entities(
        tax_benefit_system.clone(), situation_examples.single
    )


def configure_disk(simulation):
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    for name in ("salary", "income_tax"):
        holder = simulation.get_holder(name)
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True


@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
def test_parent_input_write_never_invalidates_existing_branch(simulation, disk):
    if disk:
        configure_disk(simulation)
    simulation.set_input("salary", "2017-01", [4000])
    branch = simulation.get_branch("snapshot")
    result = branch.calculate("income_tax", "2017-01")
    simulation.set_input("salary", "2017-01", [8000])
    assert branch.calculate("income_tax", "2017-01") is result
    np.testing.assert_array_equal(result, [600])
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [1200])


@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
def test_direct_holder_input_invalidates_dependent_results(simulation, disk):
    if disk:
        configure_disk(simulation)
    simulation.set_input("salary", "2017-01", [4000])
    simulation.calculate("income_tax", "2017-01")
    simulation.get_holder("salary").set_input("2017-01", [8000])
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [1200])


@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
def test_holder_delete_forgets_provenance_and_results(simulation, disk):
    if disk:
        configure_disk(simulation)
    simulation.set_input("salary", "2017-01", [4000])
    simulation.calculate("income_tax", "2017-01")
    simulation.get_holder("salary").delete_arrays("2017")
    assert simulation.supplied_input_periods("salary") == []
    assert len(simulation.result_cache) == 0
    assert simulation.get_supplied_input("salary", "2017-01") is None
    assert simulation.get_holder("income_tax").get_array(period("2017-01")) is None


@pytest.mark.parametrize(
    "value",
    [[1, 2], [float("nan")], ["not numeric"]],
    ids=["wrong-size", "nan", "wrong-dtype"],
)
def test_rejected_input_preserves_results_and_provenance(simulation, value):
    simulation.set_input("salary", "2017-01", [4000])
    result = simulation.calculate("income_tax", "2017-01")
    keys = simulation.result_cache.supplied_input_keys()
    with pytest.raises(ValueError):
        simulation.set_input("salary", "2017-01", value)
    assert simulation.calculate("income_tax", "2017-01") is result
    assert simulation.result_cache.supplied_input_keys() == keys
    np.testing.assert_array_equal(
        simulation.get_supplied_input("salary", "2017-01"), [4000]
    )


def test_neutralized_ignored_input_does_not_record_provenance(simulation):
    simulation.tax_benefit_system.neutralize_variable("salary")
    simulation.rebind_tax_benefit_system()
    with pytest.warns(Warning, match="neutralized"):
        simulation.set_input("salary", "2017-01", [4000])
    assert simulation.supplied_input_periods("salary") == []


def test_input_loading_does_not_scan_or_reinsert_prior_inputs(simulation):
    simulation.clear_calculated_results()
    holder = simulation.get_holder("salary")
    with patch.object(
        holder._memory_storage, "put", wraps=holder._memory_storage.put
    ) as put:
        with patch.object(
            holder,
            "clear_calculated_results",
            side_effect=AssertionError("input holders must not be scanned"),
        ):
            for year in range(2000, 2050):
                simulation.set_input("salary", f"{year}-01", [4000])
    assert put.call_count == 50
    assert len(simulation.supplied_input_periods("salary")) == 50


def test_retaining_inputs_is_local_to_parent(simulation):
    simulation.set_input("salary", "2017-01", [4000])
    branch = simulation.get_branch("snapshot")
    result = branch.calculate("income_tax", "2017-01")
    simulation.retain_supplied_inputs([])
    assert simulation.get_supplied_input("salary", "2017-01") is None
    np.testing.assert_array_equal(
        branch.get_supplied_input("salary", "2017-01"), [4000]
    )
    assert branch.calculate("income_tax", "2017-01") is result


def test_shared_parameter_revision_checked_before_fast_result(simulation):
    simulation.set_input("salary", "2017-01", [4000])
    branch = simulation.get_branch("shared_policy")
    isolated = simulation.get_branch("private_policy", clone_system=True)
    for target in (simulation, branch, isolated):
        np.testing.assert_array_equal(target.calculate("income_tax", "2017-01"), [600])
    simulation.tax_benefit_system.parameters.taxes.income_tax_rate.update(
        period="2017", value=0.25
    )
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [1000])
    np.testing.assert_array_equal(branch.calculate("income_tax", "2017-01"), [1000])
    np.testing.assert_array_equal(isolated.calculate("income_tax", "2017-01"), [600])


def test_supported_variable_mutation_invalidates_shared_system_results(simulation):
    simulation.set_input("salary", "2017-01", [4000])
    branch = simulation.get_branch("shared_policy")
    simulation.calculate("income_tax", "2017-01")
    branch.calculate("income_tax", "2017-01")
    simulation.tax_benefit_system.neutralize_variable("income_tax")
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [0])
    np.testing.assert_array_equal(branch.calculate("income_tax", "2017-01"), [0])


def test_supported_variable_update_rebinds_existing_holder(simulation):
    simulation.set_input("salary", "2017-01", [4000])
    simulation.calculate("income_tax", "2017-01")

    class income_tax(Variable):
        def formula(person, period):
            return person("salary", period) * 0.5

    simulation.tax_benefit_system.update_variable(income_tax)
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [2000])


def test_holder_deleting_a_year_evicts_fast_month_results(simulation):
    simulation.calculate("income_tax", "2017-01")
    simulation.calculate("income_tax", "2017-02")
    simulation.get_holder("income_tax").delete_arrays("2017")
    assert (
        ResultCacheKey("income_tax", period("2017-01")) not in simulation.result_cache
    )
    assert (
        ResultCacheKey("income_tax", period("2017-02")) not in simulation.result_cache
    )


def test_supplied_period_replacement_invalidates_cached_aggregate(simulation):
    simulation.set_input("salary", "2017-01", [4000])
    np.testing.assert_array_equal(simulation.calculate_add("salary", "2017"), [4000])
    simulation.get_holder("salary").set_input("2017-01", [8000])
    np.testing.assert_array_equal(simulation.calculate_add("salary", "2017"), [8000])


def test_low_level_cache_write_is_not_supplied_input(simulation):
    holder = simulation.get_holder("salary")
    holder.put_in_cache(np.array([4000]), period("2017-01"))
    assert simulation.supplied_input_periods("salary") == []
    simulation.clear_calculated_results()
    assert holder.get_array(period("2017-01")) is None


def test_legacy_cache_init_before_parameter_attribute():
    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

    system = object.__new__(TaxBenefitSystem)
    system._parameters_at_instant_cache = {}
    assert len(system._parameters_at_instant_cache) == 0


@pytest.mark.parametrize(
    "derived", [False, True], ids=["legacy-cache", "derived-cache"]
)
def test_cache_insertion_never_overwrites_supplied_input(simulation, derived):
    simulation.set_input("salary", "2017-01", [4000])
    holder = simulation.get_holder("salary")
    holder.put_in_cache(np.array([9000]), period("2017-01"), derived=derived)
    np.testing.assert_array_equal(
        simulation.get_supplied_input("salary", "2017-01"), [4000]
    )
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [600])


def test_rebinding_shared_system_preserves_its_existing_backreference(simulation):
    branch = simulation.get_branch("shared")
    original = branch.tax_benefit_system.simulation
    branch.rebind_tax_benefit_system()
    assert branch.tax_benefit_system.simulation is original


def test_rebinding_owned_system_sets_its_backreference_when_requested(simulation):
    branch = simulation.get_branch("private", clone_system=True)
    branch.tax_benefit_system.simulation = simulation
    branch.rebind_tax_benefit_system(set_simulation_backreference=True)
    assert branch.tax_benefit_system.simulation is branch


POLICY_MUTATIONS = ["parameter-update", "neutralize-variable", "replace-parameter-tree"]


def mutate_policy(simulation, mutation):
    system = simulation.tax_benefit_system
    if mutation == "neutralize-variable":
        system.neutralize_variable("income_tax")
        return 0
    if mutation == "replace-parameter-tree":
        system.replace_parameters(system.parameters.clone())
    system.parameters.taxes.income_tax_rate.update(period="2017", value=0.25)
    return 1000


@pytest.mark.parametrize("mutation", POLICY_MUTATIONS, ids=POLICY_MUTATIONS)
@pytest.mark.parametrize(
    "copy_mode", ["clone", "shared-clone", "branch", "private-branch"]
)
def test_copy_after_policy_mutation_cannot_accept_stale_results(
    simulation, mutation, copy_mode
):
    simulation.set_input("salary", "2017-01", [4000])
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [600])
    expected = mutate_policy(simulation, mutation)
    if copy_mode == "clone":
        copied = simulation.clone()
    elif copy_mode == "shared-clone":
        copied = simulation.clone(clone_tax_benefit_system=False)
    else:
        copied = simulation.get_branch(
            copy_mode, clone_system=copy_mode == "private-branch"
        )
    np.testing.assert_array_equal(copied.calculate("income_tax", "2017-01"), [expected])
    np.testing.assert_array_equal(
        simulation.calculate("income_tax", "2017-01"), [expected]
    )
    np.testing.assert_array_equal(
        copied.get_supplied_input("salary", "2017-01"), [4000]
    )


@pytest.mark.parametrize("mutation", POLICY_MUTATIONS, ids=POLICY_MUTATIONS)
def test_rebinding_after_policy_mutation_cannot_accept_stale_results(
    simulation, mutation
):
    simulation.set_input("salary", "2017-01", [4000])
    np.testing.assert_array_equal(simulation.calculate("income_tax", "2017-01"), [600])
    expected = mutate_policy(simulation, mutation)
    simulation.rebind_tax_benefit_system()
    assert len(simulation.result_cache) == 0
    np.testing.assert_array_equal(
        simulation.calculate("income_tax", "2017-01"), [expected]
    )


def test_supplied_periods_are_chronological_across_period_units(simulation):
    holder = simulation.get_holder("salary")
    # Direct helper-free writes exercise periods whose canonical text starts
    # with a unit name, while provenance still uses valid Period values.
    with simulation.result_cache.supplied_input_context("default"):
        holder._set(period("year:2017:2"), [100])
        holder._set(period("2019-01"), [200])
    assert simulation.supplied_input_periods("salary") == [
        period("year:2017:2"),
        period("2019-01"),
    ]


@pytest.mark.parametrize(
    "branch_name",
    ["", None, 123, "bad:name", "../bad", "bad\\name"],
    ids=["empty", "none", "non-string", "delimiter", "slash", "backslash"],
)
def test_invalid_input_branch_is_rejected_before_storage_mutation(
    simulation, branch_name
):
    holder = simulation.get_holder("salary")
    simulation.set_input("salary", "2017-01", [4000])
    result = simulation.calculate("income_tax", "2017-01")
    before = holder.get_known_branch_periods()
    revision = simulation.input_revision
    with pytest.raises(ValueError):
        holder.set_input("2017-01", [9000], branch_name)
    assert holder.get_known_branch_periods() == before
    assert simulation.input_revision == revision
    assert simulation.calculate("income_tax", "2017-01") is result


def test_input_revision_advances_for_successful_writes_and_replacements(simulation):
    revision = simulation.input_revision
    simulation.set_input("salary", "2017-01", [4000])
    assert simulation.input_revision > revision
    revision = simulation.input_revision
    simulation.get_holder("salary").set_input("2017-01", [5000])
    assert simulation.input_revision > revision


@pytest.mark.parametrize(
    "operation",
    ["calculate", "clear", "missing-delete", "rejected-write", "neutralized-write"],
)
def test_non_input_mutations_do_not_advance_input_revision(simulation, operation):
    revision = simulation.input_revision
    if operation == "calculate":
        simulation.calculate("income_tax", "2017-01")
    elif operation == "clear":
        simulation.clear_calculated_results()
    elif operation == "missing-delete":
        simulation.delete_arrays("salary", "2017-01")
    elif operation == "rejected-write":
        with pytest.raises(ValueError):
            simulation.set_input("salary", "2017-01", [float("nan")])
    else:
        simulation.tax_benefit_system.neutralize_variable("salary")
        simulation.rebind_tax_benefit_system()
        with pytest.warns(Warning):
            simulation.set_input("salary", "2017-01", [4000])
    assert simulation.input_revision == revision


@pytest.mark.parametrize("operation", ["delete", "holder-delete", "retain"])
def test_removing_supplied_inputs_advances_owner_revision_only(simulation, operation):
    simulation.set_input("salary", "2017-01", [4000])
    branch = simulation.get_branch("snapshot")
    revision = simulation.input_revision
    assert branch.input_revision == revision
    if operation == "delete":
        simulation.delete_arrays("salary", "2017-01")
    elif operation == "holder-delete":
        simulation.get_holder("salary").delete_arrays("2017-01")
    else:
        simulation.retain_supplied_inputs([])
    assert simulation.input_revision > revision
    assert branch.input_revision == revision
    np.testing.assert_array_equal(
        branch.get_supplied_input("salary", "2017-01"), [4000]
    )


def test_clone_input_revision_advances_independently(simulation):
    clone = simulation.clone()
    revision = simulation.input_revision
    assert clone.input_revision == revision
    clone.set_input("salary", "2017-01", [4000])
    assert clone.input_revision > revision
    assert simulation.input_revision == revision
