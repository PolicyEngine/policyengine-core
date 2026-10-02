"""Regression tests: the record of ``set_input`` values follows storage.

``Simulation._user_input_keys`` records each (variable, branch, period) a
simulation stored through ``set_input``. ``_invalidate_all_caches`` (run by
``apply_reform``) keeps the values it names and ``to_input_dataframe``
exports them, and country packages read it to tell a value entered directly
from one calculated by a formula.

The record used to drift from storage (policyengine-core#559):
``delete_arrays`` deleted values but kept their entries, so a value
calculated later for the same period counted as an input, and ``clone``
(so also ``get_branch``) shared one record between simulations that store
their values separately.
"""

from __future__ import annotations

import numpy as np

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.experimental import MemoryConfig
from policyengine_core.model_api import MONTH, Reform, Variable
from policyengine_core.simulations import SimulationBuilder

JANUARY = "2025-01"
FEBRUARY = "2025-02"


def _simulation(tax_benefit_system=None):
    return SimulationBuilder().build_from_entities(
        tax_benefit_system or CountryTaxBenefitSystem(),
        {
            "persons": {"bill": {"salary": {JANUARY: 1_000}}},
            "households": {"household": {"parents": ["bill"]}},
        },
    )


def _key(variable, period, branch="default"):
    return (variable, branch, periods.period(period))


def _input_periods(simulation, variable):
    return {
        (branch, str(period))
        for name, branch, period in simulation._user_input_keys
        if name == variable
    }


class income_tax(Variable):
    value_type = float
    entity = Person
    definition_period = MONTH
    label = "Income tax at twice the rate"

    def formula(person, period, parameters):
        return person("salary", period) * 0.3


class DoubleIncomeTaxRate(Reform):
    def apply(self):
        self.update_variable(income_tax)


def test_formula_result_after_delete_is_not_kept_by_apply_reform():
    """A value calculated after its input was deleted is recalculated."""
    simulation = _simulation()
    simulation.set_input("income_tax", JANUARY, [5.0])
    simulation.delete_arrays("income_tax")

    assert simulation.calculate("income_tax", JANUARY)[0] == 150.0
    assert _key("income_tax", JANUARY) not in simulation._user_input_keys

    simulation.apply_reform(DoubleIncomeTaxRate)

    assert simulation.get_holder("income_tax").get_array(JANUARY) is None
    assert simulation.calculate("income_tax", JANUARY)[0] == 300.0


def test_inputs_not_deleted_are_still_kept_by_apply_reform():
    simulation = _simulation()
    simulation.set_input("income_tax", FEBRUARY, [5.0])
    simulation.delete_arrays("income_tax", JANUARY)

    simulation.apply_reform(DoubleIncomeTaxRate)

    assert simulation.calculate("income_tax", FEBRUARY)[0] == 5.0
    assert simulation.calculate("salary", JANUARY)[0] == 1_000.0


def test_default_value_after_delete_is_not_exported():
    """An input variable read after its value was deleted gets its default,
    which ``to_input_dataframe`` must not export as an input."""
    simulation = _simulation()
    simulation.delete_arrays("salary")

    assert simulation.calculate("salary", JANUARY)[0] == 0.0
    assert "salary__2025-01" not in simulation.to_input_dataframe().columns


def test_delete_for_a_period_drops_only_entries_within_it():
    simulation = _simulation()
    for month in ("2024-12", FEBRUARY, "2025-03"):
        simulation.set_input("salary", month, [2_000.0])

    simulation.delete_arrays("salary", FEBRUARY)
    assert _input_periods(simulation, "salary") == {
        ("default", "2024-12"),
        ("default", JANUARY),
        ("default", "2025-03"),
    }

    simulation.delete_arrays("salary", "2025")
    assert _input_periods(simulation, "salary") == {("default", "2024-12")}


def test_delete_of_an_eternal_variable_drops_its_entry_for_any_period():
    """Eternal values are stored once, whatever period they were set for,
    so deleting any period deletes the value and its entry."""
    simulation = _simulation()
    simulation.set_input("birth", "2020", ["1980-01-01"])

    simulation.delete_arrays("birth", "2031-07")

    assert simulation.get_holder("birth").get_known_periods() == []
    assert _input_periods(simulation, "birth") == set()


def test_holder_delete_drops_the_entry():
    """Code that moves an input to another variable through the holder (as
    country packages do before modelling a behavioural response) leaves no
    entry for the variable it moved the value from."""
    simulation = _simulation()
    holder = simulation.get_holder("salary")
    simulation.set_input("pension", JANUARY, holder.get_array(JANUARY))
    holder.delete_arrays(JANUARY)

    assert _input_periods(simulation, "salary") == set()
    assert _key("pension", JANUARY) in simulation._user_input_keys
    assert simulation.calculate("salary", JANUARY)[0] == 0.0

    simulation.apply_reform(DoubleIncomeTaxRate)

    assert holder.get_array(JANUARY) is None
    assert "salary__2025-01" not in simulation.to_input_dataframe().columns


def test_branch_delete_drops_entries_for_the_branches_it_deletes_from():
    simulation = _simulation()
    branch = simulation.get_branch("branch")
    nested = branch.get_branch("nested")
    nested.set_input("salary", FEBRUARY, [3_000.0])
    holder = nested.get_holder("salary")
    holder._memory_storage.put(np.array([7.0]), JANUARY, "unrelated")
    nested._user_input_keys.add(_key("salary", JANUARY, "unrelated"))

    nested.delete_arrays("salary")

    assert _input_periods(nested, "salary") == {("unrelated", JANUARY)}
    assert _input_periods(branch, "salary") == {("default", JANUARY)}
    assert _input_periods(simulation, "salary") == {("default", JANUARY)}
    assert branch.calculate("salary", JANUARY)[0] == 1_000.0


def test_entry_is_kept_for_a_value_disk_storage_did_not_delete():
    """Disk storage deletes only the period asked for, not the periods
    within it, so the entry for a monthly value it still holds stays."""
    simulation = _simulation()
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder("rent")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    simulation.set_input("rent", JANUARY, [500.0])
    simulation.set_input("rent", FEBRUARY, [600.0])
    assert holder._memory_storage.get(JANUARY) is None

    simulation.delete_arrays("rent", "2025")
    assert _input_periods(simulation, "rent") == {
        ("default", JANUARY),
        ("default", FEBRUARY),
    }

    simulation.delete_arrays("rent", JANUARY)
    assert _input_periods(simulation, "rent") == {("default", FEBRUARY)}
    assert holder.get_array(FEBRUARY)[0] == 600.0


def test_clone_keeps_its_own_record():
    simulation = _simulation()
    simulation.calculate("income_tax", JANUARY)
    clone = simulation.clone()

    assert clone._user_input_keys == simulation._user_input_keys
    assert clone._user_input_keys is not simulation._user_input_keys

    clone.set_input("income_tax", JANUARY, [7.0])
    simulation.set_input("salary", FEBRUARY, [2_000.0])

    assert _key("income_tax", JANUARY) not in simulation._user_input_keys
    assert _key("salary", FEBRUARY) not in clone._user_input_keys
    assert simulation.get_holder("income_tax").get_array(JANUARY)[0] == 150.0

    simulation.apply_reform(DoubleIncomeTaxRate)
    clone.apply_reform(DoubleIncomeTaxRate)

    assert simulation.calculate("income_tax", JANUARY)[0] == 300.0
    assert clone.calculate("income_tax", JANUARY)[0] == 7.0
    assert clone.calculate("salary", JANUARY)[0] == 1_000.0


def test_branch_does_not_see_inputs_set_on_its_parent_after_it_was_created():
    simulation = _simulation()
    branch = simulation.get_branch("branch")
    simulation.set_input("salary", FEBRUARY, [2_000.0])

    assert _key("salary", FEBRUARY) not in branch._user_input_keys
    assert branch.calculate("salary", FEBRUARY)[0] == 0.0
    exported = branch.to_input_dataframe()
    assert "salary__2025-02" not in exported.columns
    assert exported["salary__2025-01"].tolist() == [1_000.0]


def test_parent_does_not_see_inputs_set_on_its_branch():
    simulation = _simulation()
    branch = simulation.get_branch("branch")
    branch.set_input("salary", FEBRUARY, [2_000.0])

    assert _key("salary", FEBRUARY, "branch") in branch._user_input_keys
    assert _input_periods(simulation, "salary") == {("default", JANUARY)}


def test_set_input_on_a_clone_does_not_make_the_original_record_inputs():
    """While ``set_input`` runs on a clone, values the original calculates
    are formula results, not inputs."""
    tax_benefit_system = CountryTaxBenefitSystem()
    simulation = _simulation(tax_benefit_system)
    clone = simulation.clone()

    def set_rent_and_calculate_on_the_original(holder, period, array):
        simulation.calculate("income_tax", JANUARY)
        for month in period.get_subperiods(periods.MONTH):
            holder._set(month, array)

    tax_benefit_system.variables[
        "rent"
    ].set_input = set_rent_and_calculate_on_the_original
    clone.set_input("rent", "2025", [500.0])

    assert _key("income_tax", JANUARY) not in simulation._user_input_keys
    assert _key("rent", JANUARY) in clone._user_input_keys
    assert _key("rent", JANUARY) not in simulation._user_input_keys
