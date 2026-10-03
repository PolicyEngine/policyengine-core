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
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template import Simulation as CountryTemplateSimulation
from policyengine_core.country_template.entities import Person
from policyengine_core.data import Dataset
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.experimental import MemoryConfig
from policyengine_core.model_api import MONTH, YEAR, Reform, Variable
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


def _store_on_disk(simulation, *variables):
    """Send every value stored from now on for ``variables`` to disk."""
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    for variable in variables:
        holder = simulation.get_holder(variable)
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True


def _held(simulation, variable, entries):
    """The (branch, period) entries whose value some storage still holds.

    Disk storage deletes only the period it is given (policyengine-core#564)
    and, for a branch, the files of every branch whose name starts with that
    branch's name and "_" (#552), so after a delete it can hold values memory
    no longer does. Tests of disk deletes compare the record with this, so
    they hold however disk storage deletes.
    """
    holder = simulation.get_holder(variable)
    return {
        (branch, period)
        for branch, period in entries
        if holder._get_array_from_storage(period, branch) is not None
    }


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
    within it (policyengine-core#564), so the entry for a monthly value it
    still holds stays."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    simulation.set_input("rent", JANUARY, [500.0])
    simulation.set_input("rent", FEBRUARY, [600.0])
    assert holder._memory_storage.get(JANUARY) is None

    simulation.delete_arrays("rent", "2025")
    assert _input_periods(simulation, "rent") == _held(
        simulation, "rent", {("default", JANUARY), ("default", FEBRUARY)}
    )

    simulation.set_input("rent", FEBRUARY, [600.0])
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


def _ended(value):
    class ended(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        end = "2012-12-31"
        label = "A formula that applies until 2012"

        def formula(person, period, parameters):
            return person.filled_array(value)

    return ended


class ReformEnded(Reform):
    def apply(self):
        self.update_variable(_ended(9.0))


def _carrying_over_simulation(on_disk):
    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.auto_carry_over_input_variables = True
    tax_benefit_system.add_variable(_ended(7.0))
    simulation = _simulation(tax_benefit_system)
    if on_disk:
        _store_on_disk(simulation, "ended")
    return simulation


@pytest.mark.parametrize("on_disk", [False, True])
def test_formula_result_for_a_deleted_input_is_not_carried_over(on_disk):
    """A value calculated for the period of a deleted input is a formula
    result. ``apply_reform`` discards it, so it is not carried over to later
    periods: after the reform the simulation gives what a simulation that
    never had the input gives (found in the review of policyengine-core#562).
    """
    simulation = _carrying_over_simulation(on_disk)
    holder = simulation.get_holder("ended")
    simulation.set_input("ended", "2012", [20.0])
    simulation.delete_arrays("ended", "2012")
    assert simulation.calculate("ended", "2012")[0] == 7.0
    assert (holder._memory_storage.get("2012") is None) == on_disk

    simulation.apply_reform(ReformEnded)

    never_had_the_input = _carrying_over_simulation(on_disk)
    never_had_the_input.apply_reform(ReformEnded)
    assert simulation.calculate("ended", "2013")[0] == 0.0
    assert never_had_the_input.calculate("ended", "2013")[0] == 0.0
    assert simulation.calculate("ended", "2012")[0] == 9.0


@pytest.mark.parametrize("on_disk", [False, True])
def test_input_that_was_not_deleted_is_carried_over_after_apply_reform(on_disk):
    simulation = _carrying_over_simulation(on_disk)
    simulation.set_input("ended", "2012", [20.0])

    simulation.apply_reform(ReformEnded)

    assert simulation.calculate("ended", "2013")[0] == 20.0
    assert simulation.calculate("ended", "2012")[0] == 20.0


def test_entry_is_kept_while_disk_still_holds_a_value_deleted_from_memory():
    """A value can be stored both in memory and on disk. Deleting a year
    deletes the months within it from memory, but disk storage deletes only
    the year itself (policyengine-core#564), so the input survives on disk
    and stays an input."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    simulation.set_input("rent", JANUARY, [500.0])
    simulation.memory_config.max_memory_occupation_pc = 101
    simulation.set_input("rent", JANUARY, [500.0])
    assert holder._memory_storage.get(JANUARY) is not None
    assert holder._disk_storage.get(JANUARY) is not None

    simulation.delete_arrays("rent", "2025")

    assert holder._memory_storage.get(JANUARY) is None
    assert _input_periods(simulation, "rent") == _held(
        simulation, "rent", {("default", JANUARY)}
    )
    if _input_periods(simulation, "rent"):
        exported = simulation.to_input_dataframe()
        assert exported["rent__2025-01"].tolist() == [500.0]
        simulation._invalidate_all_caches()
        assert holder.get_array(JANUARY)[0] == 500.0


@pytest.mark.parametrize(
    "twelve_months, stored_as, salary_month",
    [
        ("month:2025-01:12", "2025", JANUARY),
        ("month:2025-03:12", "year:2025-03", "2025-03"),
    ],
)
def test_twelve_month_input_is_recorded_as_the_year_it_is_stored_as(
    twelve_months, stored_as, salary_month
):
    """Storage keys twelve months starting on the first of a month as the
    year starting then, so deleting that year deletes the input's entry."""
    simulation = _simulation()
    simulation.set_input("salary", salary_month, [1_000.0])
    simulation.set_input("income_tax", twelve_months, [5.0])
    assert _input_periods(simulation, "income_tax") == {("default", stored_as)}

    simulation.get_holder("income_tax").delete_arrays(twelve_months)

    assert _input_periods(simulation, "income_tax") == set()
    assert simulation.calculate("income_tax", stored_as)[0] == 150.0
    simulation.apply_reform(DoubleIncomeTaxRate)
    assert simulation.calculate("income_tax", stored_as)[0] == 300.0


def test_disk_delete_reads_no_file_and_does_not_look_through_the_record(
    monkeypatch,
):
    """With disk storage configured, deleting checks which keys each storage
    still holds without loading any stored file, and without going through
    every entry of the record."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent", "income_tax", "birth", "salary")
    simulation.set_input("rent", JANUARY, [500.0])
    simulation.set_input("rent", FEBRUARY, [600.0])
    simulation.set_input("rent", "2025-03", [700.0])
    simulation.set_input("birth", "2020", ["1980-01-01"])
    simulation.calculate("income_tax", JANUARY)
    # March's rent is also stored in memory, so deleting the year deletes
    # memory's copy and leaves the one on disk.
    simulation.memory_config.max_memory_occupation_pc = 101
    simulation.set_input("rent", "2025-03", [700.0])
    simulation._user_input_keys = _RecordThatMustNotBeScanned(
        simulation._user_input_keys
        | {_key(f"other_{index}", JANUARY) for index in range(1_000)}
    )

    def no_file_reads(self, file):
        raise AssertionError(f"delete_arrays read {file}")

    monkeypatch.setattr(OnDiskStorage, "_decode_file", no_file_reads)

    simulation.delete_arrays("rent", "2025")
    simulation.delete_arrays("rent", JANUARY)
    for variable in ("income_tax", "salary", "birth"):
        simulation.delete_arrays(variable, JANUARY)
    simulation.delete_arrays("salary")

    # Any period of an eternal variable deletes its one value.
    monkeypatch.undo()
    assert set(simulation._user_input_keys) == {
        _key("rent", period, branch)
        for branch, period in _held(
            simulation, "rent", {("default", FEBRUARY), ("default", "2025-03")}
        )
    } | {_key(f"other_{index}", JANUARY) for index in range(1_000)}


def test_delete_skips_disk_files_storage_did_not_name():
    """A file ``restore`` finds in the storage directory need not be named
    after a period; deleting it removes no entry and raises nothing."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    simulation.set_input("rent", JANUARY, [500.0])
    holder._disk_storage._files["default_notes"] = "notes.npy"

    simulation.delete_arrays("rent")

    assert holder._disk_storage._files == {}
    assert _input_periods(simulation, "rent") == set()


def test_disk_delete_on_a_branch_whose_name_contains_an_underscore():
    """Disk keys join the branch name and the period with "_", and branch
    names can contain "_" (country packages use names like ``no_salt``)."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    branch = simulation.get_branch("no_salt")
    branch.set_input("rent", JANUARY, [500.0])
    branch.set_input("rent", FEBRUARY, [600.0])
    assert branch.get_holder("rent")._memory_storage.get(JANUARY, "no_salt") is None

    branch.delete_arrays("rent", JANUARY)
    assert _input_periods(branch, "rent") == {("no_salt", FEBRUARY)}

    branch.get_holder("rent").delete_arrays(None, "no_salt")
    assert _input_periods(branch, "rent") == set()


def test_entry_stays_while_memory_holds_a_value_disk_deleted_for_another_branch():
    """Deleting a branch's values from disk also deletes the files of every
    branch whose name starts with that branch's name and "_" (fixed in
    policyengine-core#552). The entry of an input memory still holds stays;
    the entry of one only disk held goes."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    holder.set_input(JANUARY, [500.0], "x_y")
    holder.set_input(FEBRUARY, [600.0], "x_y")
    simulation.memory_config.max_memory_occupation_pc = 101
    holder.set_input(JANUARY, [500.0], "x_y")
    assert holder._memory_storage.get(FEBRUARY, "x_y") is None

    holder.delete_arrays(None, "x")

    assert holder._memory_storage.get(JANUARY, "x_y")[0] == 500.0
    assert _input_periods(simulation, "rent") == _held(
        simulation, "rent", {("x_y", JANUARY), ("x_y", FEBRUARY)}
    )
