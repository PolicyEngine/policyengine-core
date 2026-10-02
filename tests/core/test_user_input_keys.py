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
import pandas as pd

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template import Simulation as CountryTemplateSimulation
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
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


def test_eternal_input_is_recorded_for_the_one_value_it_stores():
    """An eternal variable stores one value whatever period it was set for,
    so its entry names that value's period, eternity."""
    simulation = _simulation()
    simulation.set_input("birth", "2020", ["1980-01-01"])
    simulation.set_input("birth", "2025-03", ["1981-01-01"])

    assert _input_periods(simulation, "birth") == {("default", "ETERNITY")}


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

    simulation.delete_arrays("rent")
    assert _input_periods(simulation, "rent") == set()


def test_entry_is_dropped_for_an_eternal_value_deleted_from_disk():
    simulation = _simulation()
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = simulation.get_holder("birth")
    holder._disk_storage = holder.create_disk_storage()
    holder._on_disk_storable = True
    simulation.set_input("birth", "2020", ["1980-01-01"])
    assert holder._memory_storage.get("2020") is None

    simulation.delete_arrays("birth", "2031-07")

    assert holder.get_array("2020") is None
    assert _input_periods(simulation, "birth") == set()


class _RecordThatMustNotBeScanned(set):
    def __iter__(self):
        raise AssertionError("delete_arrays looked through the whole record")


def test_delete_does_not_look_through_the_whole_record():
    """Deleting reads what the holder stores, not every entry of the record,
    so its cost does not grow with the number of inputs (country packages
    delete every variable on a branch for each marginal rate)."""
    simulation = _simulation()
    simulation.set_input("salary", FEBRUARY, [2_000.0])
    simulation.calculate("income_tax", JANUARY)
    simulation._user_input_keys = _RecordThatMustNotBeScanned(
        simulation._user_input_keys
        | {_key(f"other_{index}", JANUARY) for index in range(1_000)}
    )

    for variable in ("income_tax", "salary", "rent", "birth"):
        simulation.delete_arrays(variable, JANUARY)
    simulation.delete_arrays("salary")

    assert _key("salary", JANUARY) not in simulation._user_input_keys
    assert _key("salary", FEBRUARY) not in simulation._user_input_keys
    assert len(simulation._user_input_keys) == 1_000


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


def test_set_input_on_a_clone_is_not_running_on_the_original():
    """While ``set_input`` runs on a clone, values the original stores, by
    calculating or through its holders, are not inputs of that call."""
    tax_benefit_system = CountryTaxBenefitSystem()
    simulation = _simulation(tax_benefit_system)
    clone = simulation.clone()

    def set_rent_and_store_on_the_original(holder, period, array):
        simulation.calculate("income_tax", JANUARY)
        simulation.get_holder("accommodation_size")._set(JANUARY, [80.0])
        for month in period.get_subperiods(periods.MONTH):
            holder._set(month, array)

    tax_benefit_system.variables["rent"].set_input = set_rent_and_store_on_the_original
    clone.set_input("rent", "2025", [500.0])

    assert _key("income_tax", JANUARY) not in simulation._user_input_keys
    assert _key("accommodation_size", JANUARY) not in simulation._user_input_keys
    assert _key("rent", JANUARY) in clone._user_input_keys
    assert _key("rent", JANUARY) not in simulation._user_input_keys


def test_values_a_set_input_handler_calculates_are_not_inputs():
    """A custom ``set_input`` handler may calculate other variables before it
    stores the input; those are formula results, which ``apply_reform``
    recalculates."""
    tax_benefit_system = CountryTaxBenefitSystem()
    simulation = _simulation(tax_benefit_system)

    def calculate_then_set_rent(holder, period, array):
        holder.simulation.calculate("income_tax", JANUARY)
        holder._set(period.start.period(periods.MONTH), array)

    tax_benefit_system.variables["rent"].set_input = calculate_then_set_rent
    simulation.set_input("rent", "2025", [500.0])

    assert _input_periods(simulation, "income_tax") == set()
    assert _input_periods(simulation, "rent") == {("default", JANUARY)}

    simulation.set_input("salary", JANUARY, [2_000.0])
    simulation._invalidate_all_caches()

    assert simulation.calculate("income_tax", JANUARY)[0] == 300.0


def test_input_a_handler_sets_for_a_period_string_is_recorded_as_that_period():
    tax_benefit_system = CountryTaxBenefitSystem()
    simulation = _simulation(tax_benefit_system)

    def set_january_rent(holder, period, array):
        holder._set(JANUARY, array)

    tax_benefit_system.variables["rent"].set_input = set_january_rent
    simulation.set_input("rent", "2025", [123.0])

    assert _key("rent", JANUARY) in simulation._user_input_keys
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [123.0]


def test_subsample_records_only_the_inputs_it_stores():
    data = pd.DataFrame(
        {
            "person_id__2022": [1, 2],
            "household_id__2022": [1, 2],
            "person_household_id__2022": [1, 2],
            "household_weight__2022": [1.0, 1.0],
            "salary__2022-01": [1_000.0, 2_000.0],
        }
    )
    simulation = CountryTemplateSimulation(dataset=Dataset.from_dataframe(data, "2022"))
    branch = simulation.get_branch("sample")
    branch.set_input("salary", "2022-01", [3_000.0, 4_000.0])

    branch.subsample(n=1, seed="user-input-keys", time_period="2022")

    holder = branch.get_holder("salary")
    for name, branch_name, period in branch._user_input_keys:
        assert (
            branch.get_holder(name)._memory_storage.get(period, branch_name) is not None
        )
    assert _input_periods(branch, "salary") == {("sample", "2022-01")}
    assert holder._memory_storage.get("2022-01", "sample").tolist() in (
        [3_000.0],
        [4_000.0],
    )
