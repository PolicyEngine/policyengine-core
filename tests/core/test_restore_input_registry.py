"""``restore_simulation`` restores inputs as inputs.

``apply_reform`` drops every cached value except the inputs recorded in
``_user_input_keys``. ``restore_simulation`` used to put every value back
with ``put_in_cache``, which records nothing, so ``apply_reform`` on a
restored simulation dropped its inputs too: an uprated input calculated
three years later came out as ``[0, 0]`` instead of ``[1116, 85]``.

``dump_simulation`` lists the periods of the values the simulation
calculated next to each variable's arrays (``derived_periods.txt``), and
marks every dump it writes (``__entities__/dump_format.txt``).
``restore_simulation`` restores the listed values as calculated ones and
records every other value as an input. The property test is
``test_restore_input_registry_property.py``.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.enums import EnumArray
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.uprated_inputs import NoOp, build_simulation, build_system

# The dump records storage provenance and identifies its format explicitly.
DERIVED_PERIODS_FILE = "derived_periods.txt"
DUMP_FORMAT_FILE = os.path.join("__entities__", "dump_format.txt")


@pytest.fixture(scope="module")
def system():
    return build_system()


def _dump_and_restore(simulation, directory):
    dump_simulation(simulation, str(directory))
    return restore_simulation(str(directory), simulation.tax_benefit_system)


def _inputs(simulation, variable):
    return sorted(
        str(period)
        for name, branch_name, period in simulation._user_input_keys
        if name == variable and branch_name == "default"
    )


def test_apply_reform_after_restore_keeps_restored_inputs(system, tmp_path):
    # The reported case: master gave [0, 0].
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    restored = _dump_and_restore(simulation, tmp_path)

    restored.apply_reform(NoOp)

    fresh = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    fresh.apply_reform(NoOp)
    assert fresh.calculate("uprated_count", "2015").tolist() == [1116, 85]
    assert restored.calculate("uprated_count", "2015").tolist() == [1116, 85]
    assert restored.calculate("uprated_count", "2012").tolist() == [1001, 77]


def test_restore_records_inputs_but_not_calculated_values(system, tmp_path):
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    simulation.calculate("uprated_count", "2014")
    restored = _dump_and_restore(simulation, tmp_path)

    assert _inputs(restored, "uprated_count") == ["2012"]
    # Every dumped value is back before any reform.
    for year in ("2012", "2013", "2014"):
        np.testing.assert_array_equal(
            restored.get_holder("uprated_count").get_array(periods.period(year)),
            simulation.get_holder("uprated_count").get_array(periods.period(year)),
        )

    restored.apply_reform(NoOp)

    holder = restored.get_holder("uprated_count")
    assert holder.get_array(periods.period("2013")) is None
    assert holder.get_array(periods.period("2014")) is None
    assert holder.get_array(periods.period("2012")).tolist() == [1001, 77]


def test_dump_lists_the_calculated_periods_and_marks_its_format(system, tmp_path):
    simulation = build_simulation(
        system,
        [
            ("uprated_count", "2012", [1001, 77]),
            ("uprated_count", "2014", [2001, 177]),
        ],
    )
    simulation.calculate("uprated_count", "2013")
    simulation.calculate("doubled_amount", "2013")
    dump_simulation(simulation, str(tmp_path))

    def calculated(variable):
        path = tmp_path / variable / DERIVED_PERIODS_FILE
        return sorted(path.read_text().split()) if path.exists() else []

    assert calculated("uprated_count") == ["2013"]
    # Calculated from the defaulted input, which is calculated too.
    assert calculated("doubled_amount") == ["2013"]
    assert calculated("uprated_amount") == ["2013"]
    assert (tmp_path / DUMP_FORMAT_FILE).exists()
    # Nothing else: earlier versions wrote the inputs to ``inputs.txt``.
    assert (
        sorted(
            file.name
            for variable in tmp_path.iterdir()
            if variable.name != "__entities__"
            for file in variable.iterdir()
            if file.suffix != ".npy"
        )
        == [DERIVED_PERIODS_FILE] * 3
    )


def test_restore_records_months_split_from_an_annual_input(system, tmp_path):
    # ``set_input`` helpers store the twelve months; each is an input.
    simulation = build_simulation(system, [("monthly_amount", "2013", [1200, 2400])])
    simulation.calculate("monthly_doubled", "2013-02")
    restored = _dump_and_restore(simulation, tmp_path)

    assert _inputs(restored, "monthly_amount") == sorted(
        str(month) for month in periods.period("2013").get_subperiods(periods.MONTH)
    )
    assert _inputs(restored, "monthly_doubled") == []

    restored.apply_reform(NoOp)

    assert restored.calculate("monthly_amount", "2013-07").tolist() == [100, 200]
    assert restored.get_holder("monthly_doubled").get_known_periods() == []


def test_restore_records_an_eternity_input_set_for_a_year(system, tmp_path):
    # The record names the period given to ``set_input``; storage keeps one
    # ETERNITY value, which is the input.
    simulation = build_simulation(system, [("eternal_code", "2013", [5, 6])])
    simulation.calculate("eternal_code_plus_one", "2015")
    restored = _dump_and_restore(simulation, tmp_path)

    assert len(_inputs(restored, "eternal_code")) == 1
    assert _inputs(restored, "eternal_code_plus_one") == []

    restored.apply_reform(NoOp)

    assert restored.calculate("eternal_code", "2020").tolist() == [5, 6]
    assert restored.get_holder("eternal_code_plus_one").get_known_periods() == []
    assert restored.calculate("eternal_code_plus_one", "2020").tolist() == [6, 7]


def test_an_input_set_on_a_branch_is_not_recorded_for_the_default_value(
    system, tmp_path
):
    # A branch shares its parent's input record (unless it copies it, as
    # policyengine-core#561 makes it), so its input for 2013 may be in the
    # record the dump reads. It must not make the parent's calculated 2013
    # value, which is what gets dumped, an input.
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    simulation.get_branch("reform").set_input("uprated_count", "2013", [5, 6])

    restored = _dump_and_restore(simulation, tmp_path)

    assert _inputs(restored, "uprated_count") == ["2012"]
    restored.apply_reform(NoOp)
    assert (
        restored.get_holder("uprated_count").get_array(periods.period("2013")) is None
    )


def _dump_as_written_before_3_32_16(simulation, directory):
    """A dump as policyengine-core wrote it before 3.32.16: the arrays and
    nothing else."""
    dump_simulation(simulation, str(directory))
    (directory / DUMP_FORMAT_FILE).unlink(missing_ok=True)
    for variable in os.listdir(directory):
        if variable == "__entities__":
            continue
        for file in (directory / variable).iterdir():
            if file.suffix != ".npy":
                file.unlink()


def _dump_with_derived_marks_without_a_format_marker(simulation, directory):
    """The earlier format: calculated periods listed, with no format marker."""
    dump_simulation(simulation, str(directory))
    (directory / DUMP_FORMAT_FILE).unlink(missing_ok=True)
    for variable in os.listdir(directory):
        if variable == "__entities__":
            continue
        for file in (directory / variable).iterdir():
            if file.suffix != ".npy" and file.name != DERIVED_PERIODS_FILE:
                file.unlink()


def test_restore_of_a_dump_written_before_3_32_16_keeps_every_value(system, tmp_path):
    # Those dumps say nothing about which values were calculated, so every
    # value is restored as an input.
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    _dump_as_written_before_3_32_16(simulation, tmp_path)

    with pytest.warns(UserWarning, match="does not say which of its values"):
        restored = restore_simulation(str(tmp_path), system)

    assert _inputs(restored, "uprated_count") == ["2012", "2013"]
    restored.apply_reform(NoOp)
    assert restored.calculate("uprated_count", "2012").tolist() == [1001, 77]
    np.testing.assert_array_equal(
        restored.calculate("uprated_count", "2013"),
        simulation.calculate("uprated_count", "2013"),
    )


def test_restore_of_a_dump_with_derived_marks_without_a_marker_records_its_inputs(
    system, tmp_path
):
    # Those dumps list the calculated periods, with no format marker: the
    # list says which values were inputs, so nothing warns.
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    _dump_with_derived_marks_without_a_format_marker(simulation, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = restore_simulation(str(tmp_path), system)

    assert _inputs(restored, "uprated_count") == ["2012"]
    holder = restored.get_holder("uprated_count")
    assert holder.is_derived(periods.period("2013"))
    restored.apply_reform(NoOp)
    assert holder.get_array(periods.period("2013")) is None
    assert restored.calculate("uprated_count", "2015").tolist() == [1116, 85]


def test_restore_of_a_dump_with_a_format_marker_does_not_warn(system, tmp_path):
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    dump_simulation(simulation, str(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restore_simulation(str(tmp_path), system)


def test_restore_of_a_dump_of_inputs_alone_does_not_warn(system, tmp_path):
    # It lists no calculated value, but its marker says it would have.
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    dump_simulation(simulation, str(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = restore_simulation(str(tmp_path), system)

    assert _inputs(restored, "uprated_count") == ["2012"]


def test_restored_enum_input_keeps_its_type_and_reform_provenance(system, tmp_path):
    simulation = build_simulation(
        system,
        [("housing_occupancy_status", "2013-01", ["owner", "tenant"])],
    )
    original = simulation.calculate("housing_occupancy_status", "2013-01")
    restored = _dump_and_restore(simulation, tmp_path)

    for sim in (simulation, restored):
        sim.apply_reform(NoOp)
        result = sim.calculate("housing_occupancy_status", "2013-01")
        assert isinstance(result, EnumArray)
        assert result.possible_values is original.possible_values
        assert result.dtype == original.dtype
        assert result.tobytes() == original.tobytes()
    assert _inputs(restored, "housing_occupancy_status") == ["2013-01"]


def test_a_value_cached_without_derived_is_restored_as_a_recorded_input(
    system, tmp_path
):
    # ``put_in_cache`` without ``derived=True`` stores an input: carry-over
    # and uprating read it as one, but ``set_input`` did not record it, so
    # ``apply_reform`` on the dumped simulation drops it. A dump lists which
    # values were calculated, not where an input came from, so the restored
    # simulation records it as an input, and ``apply_reform`` keeps it.
    simulation = build_simulation(system)
    simulation.get_holder("uprated_count").put_in_cache(
        np.array([1001, 77]), periods.period("2012")
    )
    assert simulation.calculate("uprated_count", "2015").tolist() == [1116, 85]
    restored = _dump_and_restore(simulation, tmp_path)

    assert _inputs(simulation, "uprated_count") == []
    assert _inputs(restored, "uprated_count") == ["2012"]
    holder = restored.get_holder("uprated_count")
    assert holder.get_input_periods() == [periods.period("2012")]
    assert restored.calculate("uprated_count", "2016").tolist() == (
        simulation.calculate("uprated_count", "2016").tolist()
    )

    simulation.apply_reform(NoOp)
    restored.apply_reform(NoOp)

    assert simulation.get_holder("uprated_count").get_known_periods() == []
    assert restored.calculate("uprated_count", "2015").tolist() == [1116, 85]


def test_a_value_calculated_after_its_input_was_deleted_is_restored_as_calculated(
    system, tmp_path
):
    # ``delete_arrays`` leaves the input record behind (policyengine-core#561
    # drops it), so the record names the value calculated there afterwards.
    # The dump lists that value as calculated, and the restored simulation
    # restores it as calculated and does not record it.
    simulation = build_simulation(
        system,
        [
            ("uprated_count", "2012", [1001, 77]),
            ("uprated_count", "2013", [5, 6]),
        ],
    )
    simulation.delete_arrays("uprated_count", "2013")
    assert simulation.calculate("uprated_count", "2013").tolist() == [1038, 79]
    restored = _dump_and_restore(simulation, tmp_path)

    assert _inputs(restored, "uprated_count") == ["2012"]
    holder = restored.get_holder("uprated_count")
    assert holder.is_derived(periods.period("2013"))
    restored.apply_reform(NoOp)
    assert holder.get_array(periods.period("2013")) is None
    assert restored.calculate("uprated_count", "2013").tolist() == [1038, 79]


def test_restored_simulation_exports_its_inputs(system, tmp_path):
    # ``to_input_dataframe`` exports what the record names: before, a
    # restored simulation exported no input variable at all.
    simulation = build_simulation(
        system,
        [
            ("uprated_count", "2012", [1001, 77]),
            ("eligible", "2012", [True, False]),
        ],
    )
    simulation.calculate("uprated_count", "2013")
    restored = _dump_and_restore(simulation, tmp_path)

    exported = restored.to_input_dataframe()
    expected = simulation.to_input_dataframe()

    assert sorted(exported.columns) == sorted(expected.columns)
    assert "uprated_count__2012" in exported.columns
    assert "uprated_count__2013" not in exported.columns
    for column in expected.columns:
        np.testing.assert_array_equal(exported[column], expected[column])
