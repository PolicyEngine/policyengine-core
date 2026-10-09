"""Value and provenance regressions for the input-record soundness review.

The replay/export record is narrower than storage's carry-over input marks:
only values written through ``set_input`` belong to that record. Two tiers
holding the same key can hold values with different input provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.model_api import Reform, Variable
from tests.core.test_user_input_keys import (
    JANUARY,
    _input_periods,
    _key,
    _simulation,
    _store_on_disk,
)


class NoopReform(Reform):
    def apply(self):
        pass


class raw_year(Variable):
    value_type = float
    entity = Person
    definition_period = periods.YEAR
    label = "Raw yearly input"


class raw_month(Variable):
    value_type = float
    entity = Person
    definition_period = periods.MONTH
    label = "Raw monthly input"


class raw_day(Variable):
    value_type = float
    entity = Person
    definition_period = periods.DAY
    label = "Raw daily input"


@pytest.mark.parametrize(
    "period_string",
    ["day:2025-01-03:2", "year:2025:2", "month:2025-01:2"],
)
def test_raw_period_keeps_its_unit_and_size_through_invalidation(period_string):
    simulation = _simulation()
    holder = simulation.get_holder("income_tax")
    # The fixture's divide handler otherwise expands a year into months and
    # converts a day period into an unsupported mid-month monthly period.
    holder.variable.set_input = None
    period = periods.period(period_string)

    holder.set_input(period, [42.0])

    assert _key("income_tax", period) in simulation._user_input_keys
    assert _input_periods(simulation, "income_tax") == {("default", period_string)}
    np.testing.assert_array_equal(holder.get_array(period), [42.0])

    simulation._invalidate_all_caches()

    assert _key("income_tax", period) in simulation._user_input_keys
    assert _input_periods(simulation, "income_tax") == {("default", period_string)}
    np.testing.assert_array_equal(holder.get_array(period), [42.0])


def test_invisible_memory_branch_deletion_removes_its_input_record():
    simulation = _simulation()
    holder = simulation.get_holder("salary")
    holder.set_input(JANUARY, [42.0], "hidden")
    assert _key("salary", JANUARY, "hidden") in simulation._user_input_keys

    holder.delete_arrays(None, "hidden")

    assert _key("salary", JANUARY, "hidden") not in simulation._user_input_keys
    assert not holder._memory_storage.has(JANUARY, "hidden")
    assert _input_periods(simulation, "salary") == {("default", JANUARY)}
    np.testing.assert_array_equal(holder.get_array(JANUARY), [1_000.0])


def test_custom_handler_preserves_its_explicit_destination_branch():
    simulation = _simulation()

    def foreign_branch_input(holder, period, array):
        holder._set(period.start.period(periods.MONTH), array, "foreign")

    simulation.tax_benefit_system.variables["rent"].set_input = foreign_branch_input
    simulation.set_input("rent", "2025", [42.0])
    holder = simulation.get_holder("rent")

    assert _input_periods(simulation, "rent") == {("foreign", JANUARY)}
    np.testing.assert_array_equal(
        holder._get_array_from_storage(JANUARY, "foreign"), [42.0]
    )
    assert holder._get_array_from_storage(JANUARY, "default") is None
    assert simulation._user_input_contexts == []


def test_disk_deletion_preserves_digits_in_the_branch_name():
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    holder.set_input(JANUARY, [42.0], "b1")
    assert _key("rent", JANUARY, "b1") in simulation._user_input_keys

    holder.delete_arrays(None, "b1")

    assert _input_periods(simulation, "rent") == set()
    assert not holder._disk_storage.has(JANUARY, "b1")


def test_branch_read_returns_a_writable_copy_without_changing_the_parent():
    simulation = _simulation()
    branch = simulation.get_branch("b1")
    array = branch.get_holder("salary").get_array(JANUARY)

    assert array.flags.writeable
    array[0] = 42.0

    np.testing.assert_array_equal(
        branch.get_holder("salary").get_array(JANUARY), [42.0]
    )
    np.testing.assert_array_equal(
        simulation.get_holder("salary").get_array(JANUARY), [1_000.0]
    )


def test_disk_input_keeps_its_value_in_storage_and_export():
    simulation = _simulation()
    _store_on_disk(simulation, "salary")
    simulation.delete_arrays("salary", JANUARY)

    simulation.set_input("salary", JANUARY, [42.0])

    assert _key("salary", JANUARY) in simulation._user_input_keys
    np.testing.assert_array_equal(
        simulation.get_holder("salary")._disk_storage.get(JANUARY), [42.0]
    )
    assert simulation.to_input_dataframe()["salary__2025-01"].tolist() == [42.0]


@pytest.mark.parametrize("invalidate", [False, True])
def test_memory_input_wins_over_a_conflicting_disk_cache(invalidate):
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    holder.put_in_cache(np.array([99.0]), JANUARY)
    simulation.memory_config.max_memory_occupation_pc = 101
    simulation.set_input("rent", JANUARY, [42.0])
    np.testing.assert_array_equal(holder._memory_storage.get(JANUARY), [42.0])
    np.testing.assert_array_equal(holder._disk_storage.get(JANUARY), [99.0])

    if invalidate:
        simulation._invalidate_all_caches()

    np.testing.assert_array_equal(holder.get_array(JANUARY), [42.0])
    assert _key("rent", JANUARY) in simulation._user_input_keys
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [42.0]
    if invalidate:
        assert holder._disk_storage.get(JANUARY) is None


@pytest.mark.parametrize("year", [1, 99, 999, 1000])
@pytest.mark.parametrize("variable_type", [raw_year, raw_month, raw_day])
@pytest.mark.parametrize("entrypoint", ["simulation", "holder"])
@pytest.mark.parametrize("on_disk", [False, True])
def test_early_year_input_is_stored_and_recorded(
    year, variable_type, entrypoint, on_disk
):
    system = CountryTaxBenefitSystem()
    system.add_variable(variable_type)
    simulation = _simulation(system)
    variable = variable_type.__name__
    holder = simulation.get_holder(variable)
    if on_disk:
        _store_on_disk(simulation, variable)
    if variable_type is raw_year:
        period = periods.period(year)
    elif variable_type is raw_month:
        period = periods.Period((periods.MONTH, periods.instant((year, 3, 1)), 1))
    else:
        period = periods.Period((periods.DAY, periods.instant((year, 3, 17)), 2))

    if entrypoint == "simulation":
        simulation.set_input(variable, period, [120.0])
    else:
        holder.set_input(period, [120.0])

    assert (variable, "default", period) in simulation._user_input_keys
    np.testing.assert_array_equal(holder.get_array(period), [120.0])
    storage = holder._disk_storage if on_disk else holder._memory_storage
    np.testing.assert_array_equal(storage.get(period), [120.0])
    assert simulation._user_input_contexts == []

    simulation.apply_reform(NoopReform)
    np.testing.assert_array_equal(storage.get(period), [120.0])
    simulation.delete_arrays(variable)
    assert not storage.has(period)
    assert (variable, "default", period) not in simulation._user_input_keys


@pytest.mark.parametrize("on_disk", [False, True])
def test_zero_padded_early_year_string_records_the_stored_period(on_disk):
    system = CountryTaxBenefitSystem()
    system.add_variable(raw_year)
    simulation = _simulation(system)
    if on_disk:
        _store_on_disk(simulation, "raw_year")

    simulation.set_input("raw_year", "0999", [123.0])

    assert _key("raw_year", periods.period(999)) in simulation._user_input_keys
    np.testing.assert_array_equal(
        simulation.get_holder("raw_year").get_array(periods.period(999)), [123.0]
    )
    assert simulation._user_input_contexts == []


def test_deleted_memory_input_does_not_turn_a_disk_calculation_into_an_input():
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    np.testing.assert_array_equal(simulation.calculate("rent", JANUARY), [0.0])
    assert _key("rent", JANUARY) not in simulation._user_input_keys
    simulation.memory_config.max_memory_occupation_pc = 101
    simulation.set_input("rent", JANUARY, [500.0])
    np.testing.assert_array_equal(holder._memory_storage.get(JANUARY), [500.0])
    np.testing.assert_array_equal(holder._disk_storage.get(JANUARY), [0.0])

    simulation.delete_arrays("rent", "2025")

    assert holder._memory_storage.get(JANUARY) is None
    assert _key("rent", JANUARY) not in simulation._user_input_keys
    assert "rent__2025-01" not in simulation.to_input_dataframe().columns

    simulation.apply_reform(NoopReform)

    assert holder.get_array(JANUARY) is None
    assert holder._disk_storage.get(JANUARY) is None
    assert _key("rent", JANUARY) not in simulation._user_input_keys


def test_deleted_memory_input_keeps_a_surviving_real_disk_input():
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    simulation.set_input("rent", JANUARY, [500.0])
    simulation.memory_config.max_memory_occupation_pc = 101
    simulation.set_input("rent", JANUARY, [700.0])
    np.testing.assert_array_equal(holder._memory_storage.get(JANUARY), [700.0])
    np.testing.assert_array_equal(holder._disk_storage.get(JANUARY), [500.0])

    simulation.delete_arrays("rent", "2025")

    assert holder._memory_storage.get(JANUARY) is None
    assert _key("rent", JANUARY) in simulation._user_input_keys
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [500.0]

    simulation.apply_reform(NoopReform)

    assert holder._memory_storage.get(JANUARY) is None
    np.testing.assert_array_equal(holder._disk_storage.get(JANUARY), [500.0])
    assert _key("rent", JANUARY) in simulation._user_input_keys
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [500.0]


@pytest.mark.parametrize("on_disk", [False, True])
def test_reform_does_not_replay_an_unrecorded_carry_over_input(on_disk):
    simulation = _simulation()
    if on_disk:
        _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    holder.put_in_cache(np.array([38.0]), JANUARY)
    # Carry-over marks intentionally include ordinary cache writes; the
    # replay/export record includes only inputs explicitly recorded by load.
    assert not holder.is_derived(JANUARY)
    assert _key("rent", JANUARY) not in simulation._user_input_keys
    assert "rent__2025-01" not in simulation.to_input_dataframe().columns

    simulation.apply_reform(NoopReform)

    assert holder.get_array(JANUARY) is None
    assert _key("rent", JANUARY) not in simulation._user_input_keys
    np.testing.assert_array_equal(
        simulation.get_holder("salary").get_array(JANUARY), [1_000.0]
    )


def test_registered_disk_input_survives_a_new_memory_input_and_clone_deletion():
    """Restoration can register an input without holder tier metadata (#576)."""
    simulation = _simulation()
    _store_on_disk(simulation, "rent")
    holder = simulation.get_holder("rent")
    holder._disk_storage.put(np.array([500.0]), JANUARY)
    simulation._user_input_keys.add(_key("rent", JANUARY))
    assert not hasattr(holder, "_user_input_storage")
    simulation.memory_config.max_memory_occupation_pc = 101

    simulation.set_input("rent", JANUARY, [700.0])
    clone = simulation.clone()
    clone.delete_arrays("rent", "2025")

    assert _key("rent", JANUARY) in clone._user_input_keys
    assert clone.to_input_dataframe()["rent__2025-01"].tolist() == [500.0]
    assert simulation.to_input_dataframe()["rent__2025-01"].tolist() == [700.0]
    simulation.delete_arrays("rent", "2025")
    for sim in (simulation, clone):
        assert _key("rent", JANUARY) in sim._user_input_keys
        assert sim.to_input_dataframe()["rent__2025-01"].tolist() == [500.0]
        sim.apply_reform(NoopReform)
        assert sim.get_holder("rent")._memory_storage.get(JANUARY) is None
        np.testing.assert_array_equal(
            sim.get_holder("rent")._disk_storage.get(JANUARY), [500.0]
        )


def test_positional_derived_argument_keeps_masters_meaning():
    simulation = _simulation()
    holder = simulation.get_holder("rent")

    holder._set(JANUARY, np.array([42.0]), "default", False, True)

    assert holder.is_derived(JANUARY)
    assert _key("rent", JANUARY) not in simulation._user_input_keys


# Smoke environments may omit dev dependencies. Only the property is skipped;
# ordinary soundness regressions above must still run in those environments.
try:
    import hypothesis
except ImportError:
    hypothesis = None


if hypothesis is not None:
    st = hypothesis.strategies

    @hypothesis.settings(max_examples=30, deadline=None)
    @hypothesis.example(
        disk_is_input=False, delete_memory=True, disk_value=0, memory_value=500
    )
    @hypothesis.example(
        disk_is_input=True, delete_memory=True, disk_value=500, memory_value=700
    )
    @hypothesis.example(
        disk_is_input=False, delete_memory=False, disk_value=99, memory_value=42
    )
    @hypothesis.given(
        disk_is_input=st.booleans(),
        delete_memory=st.booleans(),
        disk_value=st.integers(min_value=0, max_value=1_000),
        memory_value=st.integers(min_value=0, max_value=1_000),
    )
    def test_tier_provenance_replays_the_same_inputs_as_a_fresh_simulation(
        disk_is_input, delete_memory, disk_value, memory_value
    ):
        simulation = _simulation()
        _store_on_disk(simulation, "rent")
        holder = simulation.get_holder("rent")
        if disk_is_input:
            simulation.set_input("rent", JANUARY, [disk_value])
        else:
            holder.put_in_cache(np.array([float(disk_value)]), JANUARY)
        simulation.memory_config.max_memory_occupation_pc = 101
        simulation.set_input("rent", JANUARY, [memory_value])
        if delete_memory:
            simulation.delete_arrays("rent", "2025")

        expected = (
            (disk_value if disk_is_input else None) if delete_memory else memory_value
        )
        reference = _simulation()
        if expected is not None:
            reference.set_input("rent", JANUARY, [expected])

        def rent_export(sim):
            frame = sim.to_input_dataframe()
            return {
                column: frame[column].tolist()
                for column in frame.columns
                if column.startswith("rent__")
            }

        assert _input_periods(simulation, "rent") == _input_periods(reference, "rent")
        assert rent_export(simulation) == rent_export(reference)

        simulation.apply_reform(NoopReform)
        reference.apply_reform(NoopReform)

        assert _input_periods(simulation, "rent") == _input_periods(reference, "rent")
        assert rent_export(simulation) == rent_export(reference)
        if expected is None:
            assert holder.get_array(JANUARY) is None
            assert holder._disk_storage.get(JANUARY) is None
        else:
            np.testing.assert_array_equal(
                holder.get_array(JANUARY),
                reference.get_holder("rent").get_array(JANUARY),
            )
else:

    @pytest.mark.skip(reason="Hypothesis is not installed")
    def test_tier_provenance_replays_the_same_inputs_as_a_fresh_simulation():
        pass
