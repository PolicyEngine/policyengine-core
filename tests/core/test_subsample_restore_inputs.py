"""Dump/restore preserves the source inputs used to rebuild a subsample.

For the same inputs and seed, a restored simulation and one never dumped
produce the same subsample regardless of prior calculations. Cached derived
values stay derived after restore and do not become inputs to the subsample;
explicit values of formula-backed variables remain inputs.
"""

import numpy as np
import pytest

from policyengine_core.periods import period
from policyengine_core.tools.simulation_dumper import (
    dump_simulation,
    restore_simulation,
)
from tests.fixtures.subsample_inputs import DATASET_YEAR, build, stored

SEED = "restored-inputs"


def build_with_later_input(carry_over):
    simulation = build(carry_over)
    simulation.set_input("overridden", "2023", np.full(simulation.persons.count, 7.0))
    return simulation


@pytest.mark.parametrize("carry_over", [False, True])
@pytest.mark.parametrize("calculate_before", [False, True])
@pytest.mark.parametrize("n", [1, 4])
def test_restored_subsample_matches_never_dumped_inputs(
    tmp_path, carry_over, calculate_before, n
):
    fresh = build_with_later_input(carry_over)
    source = build_with_later_input(carry_over)
    if calculate_before:
        for variable, requested_period in [
            ("doubled", DATASET_YEAR),
            ("household_total", DATASET_YEAR),
            ("person_weight", DATASET_YEAR),
            ("ended", DATASET_YEAR),
            ("base", "2024"),
            ("overridden", "2024"),
        ]:
            source.calculate(variable, requested_period)

    directory = str(tmp_path / "dump")
    dump_simulation(source, directory)
    restored = restore_simulation(directory, source.tax_benefit_system)
    # The dump stores populations and holder values, not dataset metadata.
    restored.dataset = source.dataset
    restored.default_calculation_period = source.default_calculation_period

    for variable in ["base", "overridden", "household_weight"]:
        assert (variable, "default", period(DATASET_YEAR)) in restored._user_input_keys
    assert ("overridden", "default", period("2023")) in restored._user_input_keys
    if calculate_before:
        for variable in ["doubled", "household_total", "person_weight", "ended"]:
            holder = restored.get_holder(variable)
            assert holder.get_array(period(DATASET_YEAR)) is not None
            assert holder.is_derived(period(DATASET_YEAR))
            assert (
                variable,
                "default",
                period(DATASET_YEAR),
            ) not in restored._user_input_keys
        assert ("base", "default", period("2024")) not in restored._user_input_keys
        assert (
            "overridden",
            "default",
            period("2024"),
        ) not in restored._user_input_keys

    fresh.subsample(n=n, seed=SEED, time_period=DATASET_YEAR)
    restored.subsample(n=n, seed=SEED, time_period=DATASET_YEAR)
    assert stored(restored) == stored(fresh)
    assert not any(
        variable in ["doubled", "household_total", "person_weight", "ended"]
        for variable, _, _ in stored(restored)
    )
    for variable, requested_period in [
        ("base", DATASET_YEAR),
        ("person_weight", DATASET_YEAR),
        ("overridden", DATASET_YEAR),
        ("overridden", "2023"),
        ("ended", "2023"),
    ]:
        np.testing.assert_array_equal(
            restored.calculate(variable, requested_period),
            fresh.calculate(variable, requested_period),
        )

    # The old calculation cannot override the formula after new source data.
    for simulation in [fresh, restored]:
        simulation.set_input(
            "base", DATASET_YEAR, np.full(simulation.persons.count, 5.0)
        )
        np.testing.assert_array_equal(
            simulation.calculate("doubled", DATASET_YEAR),
            np.full(simulation.persons.count, 10.0),
        )
    np.testing.assert_array_equal(
        restored.calculate("household_total", DATASET_YEAR),
        fresh.calculate("household_total", DATASET_YEAR),
    )
    assert stored(restored) == stored(fresh)
