"""Formulas that read with ADD or DIVIDE compute in a Microsimulation.

``Population.__call__`` passes ``decode_enums=False`` to ``calculate_add`` or
``calculate_divide`` for every ``options=[ADD]`` or ``options=[DIVIDE]`` read
in a formula. ``Simulation.calculate_add`` and ``calculate_divide`` take that
argument, but ``Microsimulation``'s overrides did not, so in a Microsimulation
every such read raised ``TypeError: Microsimulation.calculate_add() got an
unexpected keyword argument 'decode_enums'``. A Simulation computed the same
reads.

In a Microsimulation those two methods return a ``MicroSeries``, and pandas
pairs two Series' values by index label. ``household.project`` labels each
member's value with its household's position, so combining a projected
``MicroSeries`` with another Series raises a ``ValueError`` or pairs members
with another household's value. A formula's read with an option therefore
returns a NumPy array, as in a Simulation, and one the formula can write to.
"""

import dataclasses

import numpy as np
import pytest
from microdf import MicroSeries

from policyengine_core.enums import EnumArray
from policyengine_core.model_api import ADD, DIVIDE, MONTH
from tests.fixtures.microsimulation_option_reads import (
    DATASET_YEAR,
    READERS,
    Inputs,
    Tenure,
    expected,
    run,
    simulations,
)

EXAMPLE = Inputs(
    household_of_person=(0, 0, 1),
    household_weight=(10.0, 20.0),
    monthly_pay=(
        tuple(1.0 * month for month in range(1, 13)),
        tuple(2.0 * month for month in range(1, 13)),
        tuple(5.0 * month for month in range(1, 13)),
    ),
    yearly_grant=(120.0, 240.0, 600.0),
    household_monthly_fee=((5.0,) * 12, (0.0,) * 12),
    household_yearly_grant=(12.0, 36.0),
)

EXPECTED = {
    "pay_added": [78.0, 156.0, 390.0],
    "take_home_added": [58.5, 117.0, 292.5],
    "grant_divided": [10.0, 20.0, 50.0],
    "fee_added": [60.0, 0.0],
    "pays_fee": [True, False],
    "allowance_divided": [2.0, 6.0],
    "members_pay_added": [234.0, 390.0],
    "members_grant_divided": [30.0, 50.0],
    "fee_and_pay_added": [294.0, 390.0],
    "fee_or_all_paid": [True, True],
}


def period_for(reader) -> str:
    return "2022-06" if reader.definition_period == MONTH else DATASET_YEAR


def test_numpy_reference_gives_the_hand_computed_values():
    for reader in READERS:
        np.testing.assert_array_equal(
            expected(EXAMPLE, reader.__name__), EXPECTED[reader.__name__]
        )


@pytest.mark.parametrize("reader", READERS, ids=lambda reader: reader.__name__)
def test_formula_reading_with_an_option_computes_in_a_microsimulation(reader):
    simulation, microsimulation = simulations(EXAMPLE)
    name, period = reader.__name__, period_for(reader)

    result = microsimulation.calculate(name, period, use_weights=False)

    np.testing.assert_array_equal(result, EXPECTED[name])
    np.testing.assert_array_equal(result, simulation.calculate(name, period))


def test_projected_read_pairs_each_member_with_its_own_household():
    # Two one-person households, with household 1's member listed first. A
    # projected MicroSeries would pair each person with the other household's
    # fee and return [True, False] without an error.
    inputs = Inputs(
        household_of_person=(1, 0),
        household_weight=(10.0, 20.0),
        monthly_pay=((0.0,) * 12, (0.0,) * 12),
        yearly_grant=(0.0, 0.0),
        household_monthly_fee=((0.0,) * 12, (5.0,) * 12),
        household_yearly_grant=(0.0, 0.0),
    )
    simulation, microsimulation = simulations(inputs)

    result = microsimulation.calculate(
        "fee_or_all_paid", DATASET_YEAR, use_weights=False
    )

    np.testing.assert_array_equal(expected(inputs, "fee_or_all_paid"), [False, True])
    np.testing.assert_array_equal(result, [False, True])
    np.testing.assert_array_equal(
        simulation.calculate("fee_or_all_paid", DATASET_YEAR), [False, True]
    )


def test_weighted_result_of_a_formula_reading_with_add_carries_its_weights():
    _, microsimulation = simulations(EXAMPLE)

    result = microsimulation.calculate("pays_fee", DATASET_YEAR)

    assert isinstance(result, MicroSeries)
    np.testing.assert_array_equal(result, [True, False])
    np.testing.assert_array_equal(result.weights, [10.0, 20.0])


@pytest.mark.parametrize(
    "call",
    [
        ("population", "person", "monthly_pay", DATASET_YEAR, ADD),
        ("population", "person", "monthly_take_home", DATASET_YEAR, ADD),
        ("population", "person", "yearly_grant", "2022-06", DIVIDE),
        ("population", "household", "household_monthly_fee", DATASET_YEAR, ADD),
        ("population", "household", "household_yearly_allowance", "2022-06", DIVIDE),
        ("population", "household", "monthly_pay", DATASET_YEAR, ADD),
        ("population", "household", "yearly_grant", "2022-06", DIVIDE),
    ],
    ids=lambda call: f"{call[1]}-{call[2]}-{call[4]}",
)
def test_population_read_with_an_option_matches_a_simulation(call):
    simulation, microsimulation = simulations(EXAMPLE)
    _, entity_key, variable, period, option = call

    result = microsimulation.populations[entity_key](variable, period, options=[option])

    # A NumPy array, as in a Simulation: a MicroSeries pairs values by label.
    assert type(result) is np.ndarray
    # Formulas may write to what they read.
    assert result.flags.writeable
    assert result.tolist() == run(simulation, call)


@pytest.mark.parametrize("decode_enums", [False, True])
@pytest.mark.parametrize(
    "method,variable,period",
    [
        ("calculate_add", "monthly_pay", DATASET_YEAR),
        ("calculate_divide", "yearly_grant", "2022-06"),
    ],
)
def test_microsimulation_add_and_divide_take_decode_enums(
    method, variable, period, decode_enums
):
    _, microsimulation = simulations(EXAMPLE)
    calculate = getattr(microsimulation, method)

    unweighted_result = calculate(
        variable, period, use_weights=False, decode_enums=decode_enums
    )
    weighted_result = calculate(variable, period, decode_enums=decode_enums)

    np.testing.assert_array_equal(
        unweighted_result, calculate(variable, period, use_weights=False)
    )
    np.testing.assert_array_equal(weighted_result, unweighted_result)
    np.testing.assert_array_equal(
        weighted_result.weights, microsimulation.get_weights(variable, period)
    )


def test_enum_read_with_divide_over_a_year_gives_member_names_in_a_microsimulation():
    # This records a difference that remains. Over a year,
    # Simulation.calculate_divide reads the variable with self.calculate,
    # which in a Microsimulation decodes enums by default. So a
    # Microsimulation returns member names where a Simulation returns an
    # EnumArray, and comparing the names with a member gives False for
    # everyone. No policyengine-us formula reads an Enum with ADD or DIVIDE.
    inputs = dataclasses.replace(EXAMPLE, yearly_tenure=("OWNER", "RENTER", "OWNER"))
    simulation_read, microsimulation_read = (
        each.populations["person"]("yearly_tenure", DATASET_YEAR, options=[DIVIDE])
        for each in simulations(inputs)
    )

    assert isinstance(simulation_read, EnumArray)
    assert simulation_read.decode_to_str().tolist() == ["OWNER", "RENTER", "OWNER"]
    assert (simulation_read == Tenure.OWNER).tolist() == [True, False, True]
    assert microsimulation_read.tolist() == ["OWNER", "RENTER", "OWNER"]
    assert (microsimulation_read == Tenure.OWNER).tolist() == [False, False, False]
