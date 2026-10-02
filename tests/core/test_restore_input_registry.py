"""``restore_simulation`` restores inputs as inputs.

``apply_reform`` drops every cached value except the inputs recorded in
``_user_input_keys``. ``restore_simulation`` used to put every value back
with ``put_in_cache``, which records nothing, so ``apply_reform`` on a
restored simulation dropped its inputs too: an uprated input calculated
three years later came out as ``[0, 0]`` instead of ``[1116, 85]``.

``dump_simulation`` now writes the periods whose values are inputs next to
each variable's arrays (``inputs.txt``), and ``restore_simulation`` records
exactly those as inputs. The property test is
``test_restore_input_registry_property.py``.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.uprated_inputs import NoOp, build_simulation, build_system

# Part of the dump format: policyengine-core#560 writes the same file.
INPUT_PERIODS_FILE = "inputs.txt"


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


def test_dump_writes_the_input_periods(system, tmp_path):
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

    def read(variable):
        with open(tmp_path / variable / INPUT_PERIODS_FILE) as file:
            return sorted(file.read().split())

    assert read("uprated_count") == ["2012", "2014"]
    # A calculated variable and a defaulted input record no inputs.
    assert read("doubled_amount") == []
    assert read("uprated_amount") == []


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


def _dump_without_input_record(simulation, directory):
    """A dump as earlier versions wrote it: the arrays and nothing else."""
    dump_simulation(simulation, str(directory))
    for variable in os.listdir(directory):
        if variable == "__entities__":
            continue
        for file in (directory / variable).iterdir():
            if file.suffix != ".npy":
                file.unlink()


def test_restore_of_a_dump_without_an_input_record_keeps_every_value(system, tmp_path):
    # Dumps written before inputs were recorded say nothing about which
    # values were calculated, so every value is restored as an input.
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    _dump_without_input_record(simulation, tmp_path)

    with pytest.warns(UserWarning, match="does not record which values were inputs"):
        restored = restore_simulation(str(tmp_path), system)

    assert _inputs(restored, "uprated_count") == ["2012", "2013"]
    restored.apply_reform(NoOp)
    assert restored.calculate("uprated_count", "2012").tolist() == [1001, 77]
    np.testing.assert_array_equal(
        restored.calculate("uprated_count", "2013"),
        simulation.calculate("uprated_count", "2013"),
    )


def test_restore_of_a_dump_with_an_input_record_does_not_warn(system, tmp_path):
    simulation = build_simulation(system, [("uprated_count", "2012", [1001, 77])])
    simulation.calculate("uprated_count", "2013")
    dump_simulation(simulation, str(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restore_simulation(str(tmp_path), system)


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
