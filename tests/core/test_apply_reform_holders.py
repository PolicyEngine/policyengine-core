"""Reforms refresh existing holders without discarding compatible inputs."""

import warnings

import numpy as np
import pytest

from policyengine_core.errors import VariableNotFoundError
from policyengine_core.parameters import ParameterNode
from policyengine_core.periods import ETERNITY, YEAR, period
from policyengine_core.reforms import Reform
from policyengine_core.variables import Variable
from tests.fixtures.reform_holders import (
    Household,
    Person,
    award,
    build,
    change_default,
    store_on_disk,
)

PERIOD = "2025"


def assert_bound(simulation):
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            assert holder.variable is simulation.tax_benefit_system.variables[name]
            assert holder.variable.entity.key == population.entity.key
            assert holder._memory_storage.is_eternal == (
                holder.variable.definition_period == ETERNITY
            )


def test_defined_for_default_does_not_depend_on_other_people():
    results = []
    for mask in ([False], [False, True]):
        simulation = build(mask)
        old_holder = simulation.get_holder("award")
        simulation.apply_reform(change_default(20))
        results.append(simulation.calculate("award", PERIOD).tolist())
        assert simulation.get_holder("award") is old_holder

    assert results == [[20], [20, 100]]


@pytest.mark.parametrize("on_disk", [False, True])
def test_replaced_variable_keeps_recorded_inputs(on_disk):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "award")
    simulation.set_input("award", PERIOD, [7, 9])
    holder = simulation.get_holder("award")
    records = simulation._user_input_keys.copy()
    simulation.input_variables = ["award", "eligible"]

    simulation.apply_reform(change_default(20))

    assert simulation.get_holder("award") is holder
    assert_bound(simulation)
    assert simulation._user_input_keys == records
    assert simulation.input_variables == ["award", "eligible"]
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [7, 9])
    np.testing.assert_array_equal(holder._get_array_from_storage(PERIOD), [7, 9])


class NeutralizeAward(Reform):
    def apply(self):
        self.neutralize_variable("award")


@pytest.mark.parametrize("on_disk", [False, True])
def test_neutralized_variable_keeps_input_storage_but_reads_its_default(on_disk):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "award")
    simulation.set_input("award", PERIOD, [7, 9])
    records = simulation._user_input_keys.copy()

    simulation.apply_reform(NeutralizeAward)

    holder = simulation.get_holder("award")
    assert_bound(simulation)
    assert holder.variable.is_neutralized
    assert simulation._user_input_keys == records
    np.testing.assert_array_equal(holder._get_array_from_storage(PERIOD), [7, 9])
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [0, 0])


class RestoreAward(Reform):
    def apply(self):
        self.replace_variable(award)


def restore_and_set_award(values):
    """Return a reform that restores the award, then sets and reads its input."""

    class RestoreAndSetAward(Reform):
        def apply(self):
            self.replace_variable(award)
            self.simulation.set_input("award", PERIOD, values)
            np.testing.assert_array_equal(
                self.simulation.calculate("award", PERIOD), values
            )

    return RestoreAndSetAward


def neutralize_and_set_award(values):
    """Return a reform that neutralizes the award, then sets its input."""

    class NeutralizeAndSetAward(Reform):
        def apply(self):
            self.neutralize_variable("award")
            self.simulation.set_input("award", PERIOD, values)

    return NeutralizeAndSetAward


def neutralized_input_warnings(caught):
    return [warning for warning in caught if "neutralized" in str(warning.message)]


@pytest.mark.parametrize("on_disk", [False, True])
def test_reform_keeps_an_input_it_sets_after_restoring_a_neutralized_variable(
    on_disk,
):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "award")
    simulation.set_input("award", PERIOD, [7, 9])
    simulation.apply_reform(NeutralizeAward)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation.apply_reform(restore_and_set_award([20, 30]))

    assert not neutralized_input_warnings(caught)
    assert_bound(simulation)
    holder = simulation.get_holder("award")
    np.testing.assert_array_equal(holder._get_array_from_storage(PERIOD), [20, 30])
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [20, 30])


@pytest.mark.parametrize("on_disk", [False, True])
def test_reform_ignores_an_input_it_sets_after_neutralizing_the_variable(on_disk):
    # A holder made after the neutralization ignores the input, so one made
    # before it does too.
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "award")
    simulation.set_input("award", PERIOD, [7, 9])

    with pytest.warns(Warning, match="neutralized"):
        simulation.apply_reform(neutralize_and_set_award([1, 2]))

    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [0, 0])
    simulation.apply_reform(RestoreAward)
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [7, 9])


def test_input_set_while_a_variable_is_neutralized_is_ignored():
    simulation = build()
    simulation.set_input("award", PERIOD, [7, 9])
    simulation.apply_reform(NeutralizeAward)

    with pytest.warns(Warning, match="neutralized"):
        simulation.set_input("award", PERIOD, [55, 55])

    simulation.apply_reform(RestoreAward)
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [7, 9])


def test_holders_follow_a_branch_system_changed_outside_a_reform():
    simulation = build()
    simulation.set_input("award", PERIOD, [7, 9])
    branch = simulation.get_branch("measured", clone_system=True)

    branch.tax_benefit_system.neutralize_variable("award")

    assert branch.get_holder("award").variable.is_neutralized
    assert not simulation.get_holder("award").variable.is_neutralized
    np.testing.assert_array_equal(branch.calculate("award", PERIOD), [0, 0])
    with pytest.warns(Warning, match="neutralized"):
        branch.set_input("award", PERIOD, [1, 1])
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [7, 9])


def test_reform_that_drops_no_input_keeps_each_input_list():
    simulation = build()
    simulation.set_input("income", PERIOD, [7, 9])
    names = simulation.input_variables = ["income", "eligible"]
    branch = simulation.get_branch("other")
    branch_names = branch.input_variables

    simulation.apply_reform(change_default(20))

    # Callers may hold either list, and a branch may share its parent's.
    assert simulation.input_variables is names
    assert branch.input_variables is branch_names
    assert names == ["income", "eligible"]


class added(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    default_value = 13
    label = "Added amount"


class AddAndRemove(Reform):
    def apply(self):
        self.add_variable(added)
        del self.variables["annual_value"]


@pytest.mark.parametrize("on_disk", [False, True])
def test_added_variable_is_lazy_and_removed_variable_drops_holder_and_inputs(on_disk):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "annual_value")
    simulation.set_input("annual_value", PERIOD, [7, 9])
    simulation.input_variables = ["annual_value", "eligible"]

    names = simulation.input_variables

    simulation.apply_reform(AddAndRemove)

    assert_bound(simulation)
    assert "annual_value" not in simulation.persons._holders
    assert "annual_value" not in simulation.input_variables
    # The list may be shared, so the reform replaces it instead of editing it.
    assert names == ["annual_value", "eligible"]
    assert all(key[0] != "annual_value" for key in simulation._user_input_keys)
    with pytest.raises(VariableNotFoundError):
        simulation.get_holder("annual_value")
    assert "added" not in simulation.persons._holders
    np.testing.assert_array_equal(simulation.calculate("added", PERIOD), [13, 13])
    assert_bound(simulation)


def test_tuple_reform_rebinds_after_each_reform():
    class CheckFirstReform(Reform):
        def apply(self):
            assert (
                self.simulation.get_holder("award").variable is self.variables["award"]
            )
            change_default(30).apply(self)

    simulation = build([False])
    simulation.apply_reform((change_default(20), CheckFirstReform))

    assert_bound(simulation)
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [30])


def test_dict_reform_keeps_inputs_and_rebinds_stale_holders():
    simulation = build([False])
    simulation.tax_benefit_system.parameters = ParameterNode(
        "", data={"amount": {"values": {"2025-01-01": {"value": 1}}}}
    )
    simulation.set_input("income", PERIOD, [7])
    change_default(20).apply(simulation.tax_benefit_system)

    simulation.apply_reform({"amount": {"2025": 2}})

    assert_bound(simulation)
    assert simulation.tax_benefit_system.parameters("2025-01-01").amount == 2
    np.testing.assert_array_equal(simulation.calculate("income", PERIOD), [7])
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [20])


def test_existing_baseline_keeps_its_own_system():
    simulation = build([False])
    baseline = simulation.get_branch("baseline", clone_system=True)
    simulation.baseline = baseline
    baseline_system = baseline.tax_benefit_system
    np.testing.assert_array_equal(baseline.calculate("award", PERIOD), [0])

    simulation.apply_reform(change_default(20))

    assert simulation.baseline is baseline
    assert baseline.tax_benefit_system is baseline_system
    assert_bound(simulation)
    assert_bound(baseline)
    np.testing.assert_array_equal(baseline.calculate("award", PERIOD), [0])
    np.testing.assert_array_equal(simulation.calculate("award", PERIOD), [20])


def test_existing_branches_rebind_to_their_own_systems():
    simulation = build([False])
    shared = simulation.get_branch("shared")
    nested = shared.get_branch("nested")
    independent = simulation.get_branch("independent", clone_system=True)
    independent.apply_reform(change_default(40))
    for branch in (simulation, shared, nested, independent):
        branch.set_input("income", PERIOD, [7])
        branch.calculate("award", PERIOD)

    simulation.apply_reform(change_default(20))

    for branch, expected in (
        (simulation, 20),
        (shared, 20),
        (nested, 20),
        (independent, 40),
    ):
        assert_bound(branch)
        np.testing.assert_array_equal(branch.calculate("award", PERIOD), [expected])
        np.testing.assert_array_equal(branch.calculate("income", PERIOD), [7])


class MoveAnnualToHousehold(Reform):
    def apply(self):
        class annual_value(Variable):
            entity = Household
            default_value = 17

        self.update_variable(annual_value)


def test_entity_change_drops_orphaned_holder_and_input_records():
    simulation = build()
    simulation.set_input("annual_value", PERIOD, [7, 9])
    simulation.input_variables = ["annual_value", "eligible"]

    simulation.apply_reform(MoveAnnualToHousehold)

    assert "annual_value" not in simulation.persons._holders
    assert "annual_value" not in simulation.input_variables
    assert all(key[0] != "annual_value" for key in simulation._user_input_keys)
    np.testing.assert_array_equal(simulation.calculate("annual_value", PERIOD), [17])
    assert_bound(simulation)


@pytest.mark.parametrize("on_disk", [False, True])
def test_entity_change_keeps_inputs_supplied_to_the_new_holder(on_disk):
    class MoveAndSupply(Reform):
        def apply(self):
            MoveAnnualToHousehold.apply(self)
            self.simulation.set_input("annual_value", PERIOD, [31])

    simulation = build()
    if on_disk:
        store_on_disk(simulation, "annual_value")
    simulation.set_input("annual_value", PERIOD, [7, 9])
    simulation.input_variables = ["annual_value", "eligible"]

    simulation.apply_reform(MoveAndSupply)

    assert "annual_value" not in simulation.persons._holders
    assert "annual_value" in simulation.input_variables
    assert ("annual_value", "default", period(PERIOD)) in simulation._user_input_keys
    np.testing.assert_array_equal(simulation.calculate("annual_value", PERIOD), [31])
    assert_bound(simulation)


class MakeAnnual(Reform):
    def apply(self):
        class eternal_value(Variable):
            definition_period = YEAR

            def formula(person, period):
                return person.filled_array(period.start.year)

        self.update_variable(eternal_value)


class MakeEternal(Reform):
    def apply(self):
        class annual_value(Variable):
            definition_period = ETERNITY

        self.update_variable(annual_value)


@pytest.mark.parametrize("on_disk", [False, True])
def test_eternity_to_year_drops_incompatible_storage_and_inputs(on_disk):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "eternal_value")
    simulation.set_input("eternal_value", PERIOD, [7, 9])
    simulation.input_variables = ["eternal_value", "eligible"]

    simulation.apply_reform(MakeAnnual)

    np.testing.assert_array_equal(
        simulation.calculate("eternal_value", "2025"), [2025, 2025]
    )
    np.testing.assert_array_equal(
        simulation.calculate("eternal_value", "2026"), [2026, 2026]
    )
    assert "eternal_value" not in simulation.input_variables
    assert all(key[0] != "eternal_value" for key in simulation._user_input_keys)
    assert_bound(simulation)


@pytest.mark.parametrize("on_disk", [False, True])
def test_year_to_eternity_uses_eternal_storage_for_new_inputs(on_disk):
    simulation = build()
    if on_disk:
        store_on_disk(simulation, "annual_value")
    simulation.set_input("annual_value", PERIOD, [7, 9])
    simulation.apply_reform(MakeEternal)

    assert all(key[0] != "annual_value" for key in simulation._user_input_keys)
    simulation.set_input("annual_value", PERIOD, [11, 12])

    assert_bound(simulation)
    np.testing.assert_array_equal(
        simulation.calculate("annual_value", "2025"), [11, 12]
    )
    np.testing.assert_array_equal(
        simulation.calculate("annual_value", "2026"), [11, 12]
    )
