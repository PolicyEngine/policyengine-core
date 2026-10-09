"""Unavailable sampling periods use recorded weights rather than defaults."""

from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from policyengine_core.country_template import Microsimulation, Simulation
from policyengine_core.country_template.entities import Household
from policyengine_core.data import Dataset
from policyengine_core.parameters import Parameter
from policyengine_core.periods import MONTH, YEAR
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import DATASET_YEAR, build, dataframe, stored


class household_weight(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    default_value = 1
    uprating = "weight_index"


def make_simulation(simulation_class, uprated, carry_over):
    system = build(carry_over=carry_over).tax_benefit_system
    if uprated:
        system.update_variable(household_weight)
        system.parameters.add_child(
            "weight_index",
            Parameter(
                "weight_index",
                data={"values": {"2021-01-01": 1, "2022-01-01": 2, "2023-01-01": 3}},
            ),
        )
    return simulation_class(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(dataframe(), DATASET_YEAR),
    )


def check_fallback(simulation_class, uprated, carry_over, year, n, seed):
    fresh = make_simulation(simulation_class, uprated, carry_over)
    used = make_simulation(simulation_class, uprated, carry_over)
    expected_weights = np.asarray(fresh.calculate("household_weight", DATASET_YEAR))
    expected_ids = fresh.calculate("household_id", DATASET_YEAR)
    used.calculate("household_weight", year)
    for simulation in (fresh, used):
        with patch("numpy.random.choice", wraps=np.random.choice) as choose:
            simulation.subsample(n=n, seed=seed, time_period=year)
        args, kwargs = choose.call_args
        np.testing.assert_array_equal(args[0], expected_ids)
        np.testing.assert_allclose(
            kwargs["p"], expected_weights / expected_weights.sum()
        )
        assert f"household_weight__{year}" not in simulation.dataset.load()
        assert simulation.default_calculation_period == DATASET_YEAR
    assert stored(fresh) == stored(used)


@pytest.mark.parametrize("simulation_class", [Simulation, Microsimulation])
@pytest.mark.parametrize(
    "uprated,carry_over,year",
    [
        (False, False, "2023"),
        (False, False, "2023-07"),
        (False, True, "2021"),
        (True, False, "2021"),
        (True, True, "2021-07"),
    ],
)
def test_missing_period_weights_use_dataset_input(
    simulation_class, uprated, carry_over, year
):
    check_fallback(simulation_class, uprated, carry_over, year, 3, "fallback")


@settings(max_examples=25, deadline=None)
@given(
    simulation_class=st.sampled_from([Simulation, Microsimulation]),
    case=st.sampled_from(
        [(False, False, "2023"), (False, True, "2021"), (True, True, "2021-07")]
    ),
    n=st.integers(1, 8),
    seed=st.integers(0, 100),
)
def test_weight_fallback_is_independent_of_prior_default_calculation(
    simulation_class, case, n, seed
):
    check_fallback(simulation_class, *case, n, seed)


def test_available_requested_input_takes_precedence_over_dataset_weights():
    simulation = make_simulation(Simulation, False, False)
    weights = np.arange(1, simulation.household.count + 1, dtype=float)[::-1]
    simulation.set_input("household_weight", "2023", weights)
    with patch("numpy.random.choice", wraps=np.random.choice) as choose:
        simulation.subsample(n=3, seed="explicit", time_period="2023")
    np.testing.assert_allclose(choose.call_args.kwargs["p"], weights / weights.sum())


def test_absent_weight_source_has_clear_error():
    simulation = make_simulation(Simulation, False, False)
    simulation.delete_arrays("household_weight")
    with pytest.raises(ValueError, match="household_weight.*2023.*2022"):
        simulation.subsample(n=1, seed="missing", time_period="2023")


def test_annual_weight_fallback_normalizes_dataset_period():
    simulation = make_simulation(Simulation, False, False)
    simulation.dataset.time_period = "2022-01"
    weights = np.asarray(simulation.calculate("household_weight", DATASET_YEAR))
    with patch("numpy.random.choice", wraps=np.random.choice) as choose:
        simulation.subsample(n=3, seed="dataset-month", time_period="2023")
    np.testing.assert_allclose(choose.call_args.kwargs["p"], weights / weights.sum())


def test_annual_request_uses_monthly_formula_starting_midyear():
    class monthly_weight(Variable):
        value_type = float
        entity = Household
        definition_period = MONTH

        def formula_2023_07(household, period):
            return household.filled_array(2.0)

    monthly_weight.__name__ = "household_weight"
    system = build(carry_over=False).tax_benefit_system
    system.replace_variable(monthly_weight)
    data = dataframe().rename(
        columns={"household_weight__2022": "household_weight__2022-01"}
    )
    simulation = Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(data, DATASET_YEAR),
    )
    with patch("numpy.random.choice", wraps=np.random.choice) as choose:
        simulation.subsample(n=3, seed="monthly-formula", time_period="2023")
    np.testing.assert_allclose(choose.call_args.kwargs["p"], np.full(6, 1 / 6))
