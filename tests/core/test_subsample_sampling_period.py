"""Requested-period weights select the sample without becoming source inputs."""

from unittest.mock import patch

import numpy as np
import pytest

from policyengine_core.country_template import Microsimulation, Simulation
from policyengine_core.country_template.entities import Household
from policyengine_core.data import Dataset
from policyengine_core.periods import YEAR
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import DATASET_YEAR, build, dataframe, stored


class household_weight(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Weights growing at a different rate in each household"

    def formula(household, period):
        growth = household("household_id", period) % 3 + 1
        return household("household_weight", period.last_year) * growth


def make_simulation(simulation_class):
    system = build().tax_benefit_system
    system.update_variable(household_weight)
    data = dataframe()
    # Microsimulation's export calculates weights as a side effect of
    # reading this input, reproducing the s17 variable-order case.
    data["base__2023"] = data[f"base__{DATASET_YEAR}"] + 1
    return simulation_class(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(data, DATASET_YEAR),
    )


def check_sampling_period(simulation_class, year, n, seed, quantize_weights):
    reference = make_simulation(simulation_class)
    expected_ids = np.asarray(reference.calculate("household_id", DATASET_YEAR))
    expected_weights = np.asarray(reference.calculate("household_weight", year))
    expected_probabilities = expected_weights / expected_weights.sum()
    fresh = make_simulation(simulation_class)
    used = make_simulation(simulation_class)
    used.calculate("household_weight", year)
    used.calculate("person_weight", year)

    for simulation in (fresh, used):
        original_weight_total = np.asarray(
            simulation.calculate("household_weight", DATASET_YEAR)
        ).sum()
        with patch("numpy.random.choice", wraps=np.random.choice) as choose:
            simulation.subsample(
                n=n, seed=seed, time_period=year, quantize_weights=quantize_weights
            )
        args, kwargs = choose.call_args
        np.testing.assert_array_equal(args[0], expected_ids)
        if quantize_weights:
            np.testing.assert_allclose(kwargs["p"], expected_probabilities)
        else:
            assert kwargs["p"] is None
        data = simulation.dataset.load()
        assert np.asarray(
            simulation.calculate("household_weight", DATASET_YEAR)
        ).sum() == pytest.approx(original_weight_total)
        if year != DATASET_YEAR:
            assert f"household_weight__{year}" not in data
        assert not any(name == "person_weight" for name, _, _ in stored(simulation))
        assert simulation.default_calculation_period == DATASET_YEAR

    assert stored(used) == stored(fresh)
    for variable in ("household_weight", "person_weight"):
        np.testing.assert_array_equal(
            used.calculate(variable, year), fresh.calculate(variable, year)
        )
        if year != DATASET_YEAR:
            assert used.get_holder(variable).is_derived(year)


@pytest.mark.parametrize("simulation_class", [Simulation, Microsimulation])
@pytest.mark.parametrize("year", [DATASET_YEAR, "2023", "2024", "2022-07", "2023-07"])
@pytest.mark.parametrize("quantize_weights", [False, True])
def test_sampling_uses_requested_period_weights_independently_of_calculations(
    simulation_class, year, quantize_weights
):
    check_sampling_period(simulation_class, year, 6, "probe", quantize_weights)


def test_sampling_period_is_preserved_after_repeated_subsamples():
    simulation = make_simulation(Simulation)
    for n in (6, 3):
        simulation.subsample(n=n, seed="again", time_period="2024")
        assert simulation.default_calculation_period == DATASET_YEAR
        assert "household_weight__2024" not in simulation.dataset.load()
        assert not any(
            name == "household_weight" and period == "2024"
            for name, _, period in stored(simulation)
        )


def test_monthly_sampling_uses_existing_annual_flow_conversion():
    simulation = make_simulation(Simulation)
    annual = np.asarray(simulation.calculate("household_weight", "2023"))
    monthly = np.asarray(simulation.calculate("household_weight", "2023-07"))
    np.testing.assert_allclose(monthly, annual / 12)
    np.testing.assert_allclose(monthly / monthly.sum(), annual / annual.sum())
