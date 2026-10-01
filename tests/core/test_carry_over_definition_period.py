"""Auto-carry-over reads only values stored at the variable's definition period.

A YEAR variable also caches month values: ``calculate_divide`` stores a
twelfth of the year's value, and the STOCK path stores the whole value. A
MONTH variable caches year values: ``calculate_add`` stores the sum of its
months. Since 3.24.0 auto-carry-over took the latest-starting known period
of any unit, so once any month of 2024 had been requested, a YEAR input
known only for 2024 carried a twelfth of its value into 2025.

policyengine-us computes ``monthly_age`` from ``age`` one month at a time.
A microsimulation that calculated 2024 and then 2025 on a single-year
dataset therefore aged every person to a twelfth of their age in 2025. On a
3,000-household Enhanced CPS subsample, 2025 federal income tax came to
$31.6bn, against $1,846.9bn in a simulation that calculated only 2025.

The invariant pinned here: what a simulation calculates for a later period
does not depend on which earlier periods it calculated first. Each case
compares against a fresh simulation that calculates only the later period.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import (
    CountryTaxBenefitSystem,
    entities,
    situation_examples,
)
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import QuantityType, Variable

BASE_YEAR = 2024
MONTHS = [f"{BASE_YEAR}-{month:02d}" for month in range(1, 13)]


class carried_flow(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly flow input with no formula and no uprating"


class carried_stock(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    quantity_type = QuantityType.STOCK
    label = "Yearly stock input with no formula and no uprating"


class carried_count(Variable):
    value_type = int
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly integer input with no formula and no uprating"


class carried_flag(Variable):
    value_type = bool
    entity = entities.Person
    definition_period = periods.YEAR
    label = "Yearly boolean input with no formula and no uprating"


class carried_monthly_flow(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    label = "Monthly flow input with no formula and no uprating"


YEAR_INPUTS = {
    "carried_flow": np.array([40.0, 6.0]),
    "carried_stock": np.array([40.0, 6.0]),
    "carried_count": np.array([40, 6]),
    "carried_flag": np.array([True, False]),
}


@pytest.fixture(scope="module")
def system():
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = True
    system.add_variables(
        carried_flow,
        carried_stock,
        carried_count,
        carried_flag,
        carried_monthly_flow,
    )
    return system


def _simulation(system, inputs):
    simulation = SimulationBuilder().build_from_entities(
        system, situation_examples.couple
    )
    for variable, values in inputs.items():
        for period, array in values.items():
            simulation.set_input(variable, period, array)
    return simulation


# Earlier requests that leave month (or year) caches behind before a later
# period is calculated.
EARLIER_REQUESTS = [
    [],
    [str(BASE_YEAR)],
    [MONTHS[0]],
    [MONTHS[5]],
    [MONTHS[-1]],
    MONTHS,
    [MONTHS[-1], str(BASE_YEAR)],
    [str(BASE_YEAR), *MONTHS],
]
LATER_PERIODS = ["2025", "2027", "2025-03", "2027-12"]


def test_month_cache_of_year_input_does_not_carry_into_next_year(system):
    """The policyengine-us ``monthly_age`` sequence, on a synthetic variable."""
    simulation = _simulation(system, {"carried_flow": {BASE_YEAR: [40.0, 6.0]}})

    december = simulation.calculate("carried_flow", MONTHS[-1])
    np.testing.assert_allclose(december, [40.0 / 12, 6.0 / 12])

    np.testing.assert_array_equal(
        simulation.calculate("carried_flow", BASE_YEAR + 1), [40.0, 6.0]
    )


@pytest.mark.parametrize("later_period", LATER_PERIODS)
@pytest.mark.parametrize(
    "earlier", EARLIER_REQUESTS, ids=lambda e: "+".join(e) or "none"
)
@pytest.mark.parametrize("variable", sorted(YEAR_INPUTS))
def test_year_input_later_period_does_not_depend_on_earlier_requests(
    system, variable, earlier, later_period
):
    inputs = {variable: {BASE_YEAR: YEAR_INPUTS[variable]}}
    fresh = _simulation(system, inputs).calculate(variable, later_period)

    simulation = _simulation(system, inputs)
    for period in earlier:
        simulation.calculate(variable, period)
    result = simulation.calculate(variable, later_period)

    np.testing.assert_array_equal(result, fresh)
    if periods.period(later_period).unit == periods.YEAR:
        # Carry-over of an input is the input itself.
        np.testing.assert_array_equal(result, YEAR_INPUTS[variable])


@pytest.mark.parametrize("later_period", LATER_PERIODS)
@pytest.mark.parametrize(
    "earlier", EARLIER_REQUESTS, ids=lambda e: "+".join(e) or "none"
)
def test_year_input_branch_does_not_depend_on_earlier_requests(
    system, earlier, later_period
):
    """A branch forked after the earlier requests clones their month caches."""
    inputs = {"carried_flow": {BASE_YEAR: YEAR_INPUTS["carried_flow"]}}
    fresh = _simulation(system, inputs).calculate("carried_flow", later_period)

    simulation = _simulation(system, inputs)
    for period in earlier:
        simulation.calculate("carried_flow", period)
    branch = simulation.get_branch("itemizing")

    np.testing.assert_array_equal(branch.calculate("carried_flow", later_period), fresh)
    np.testing.assert_array_equal(
        simulation.calculate("carried_flow", later_period), fresh
    )


MONTHLY_INPUT_PATTERNS = {
    "every_month": {month: [float(i + 1), 100.0 + i] for i, month in enumerate(MONTHS)},
    "march_only": {MONTHS[2]: [3.0, 103.0]},
    "december_only": {MONTHS[-1]: [12.0, 112.0]},
}


@pytest.mark.parametrize("later_period", ["2025-01", "2025-07", "2026"])
@pytest.mark.parametrize(
    "earlier", EARLIER_REQUESTS, ids=lambda e: "+".join(e) or "none"
)
@pytest.mark.parametrize("pattern", sorted(MONTHLY_INPUT_PATTERNS))
def test_month_input_later_period_does_not_depend_on_earlier_requests(
    system, pattern, earlier, later_period
):
    """A year request caches the sum of the months; it must not carry over."""
    inputs = {"carried_monthly_flow": MONTHLY_INPUT_PATTERNS[pattern]}
    fresh = _simulation(system, inputs).calculate("carried_monthly_flow", later_period)

    simulation = _simulation(system, inputs)
    for period in earlier:
        simulation.calculate("carried_monthly_flow", period)

    np.testing.assert_array_equal(
        simulation.calculate("carried_monthly_flow", later_period), fresh
    )


def test_month_input_carries_latest_month_not_year_sum(system):
    simulation = _simulation(
        system, {"carried_monthly_flow": MONTHLY_INPUT_PATTERNS["every_month"]}
    )
    np.testing.assert_array_equal(
        simulation.calculate("carried_monthly_flow", BASE_YEAR),
        [sum(range(1, 13)), sum(100.0 + i for i in range(12))],
    )
    np.testing.assert_array_equal(
        simulation.calculate("carried_monthly_flow", "2025-02"), [12.0, 111.0]
    )
