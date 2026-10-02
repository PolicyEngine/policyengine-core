"""Auto-carry-over prefers values at the variable's own definition period.

A YEAR flow variable requested for a month caches a twelfth of the year's
value at that month (``calculate_divide``), and a MONTH flow variable
requested for a year caches the sum of its months (``calculate_add``). STOCK
variables cache nothing in another unit. Since 3.24.0 auto-carry-over took
the latest-starting known period of any unit, so once a month of 2024 after
January had been requested, a YEAR flow input known only for 2024 carried a
twelfth of its value into 2025.

policyengine-us computes ``monthly_age`` from ``age`` one month at a time.
A microsimulation that calculated 2024 and then 2025 on a single-year
dataset therefore aged every person to a twelfth of their age in 2025. On a
3,000-household Enhanced CPS subsample, 2025 federal income tax came to
$31.6bn, against $1,846.9bn in a simulation that calculated only 2025.

Both caches coexist with the value they were derived from, at the
variable's own unit, so carry-over now picks the latest own-unit period and
uses another unit only when the variable has nothing at its own unit. That
keeps inputs that exist only at another unit carrying over: a variable with
no ``set_input`` helper, and every DAY variable, stores an input at whatever
period it is given. Where such a variable holds both a value at its own unit
and an input at another unit, the own-unit value now wins, as it did before
3.24.0.

The invariant pinned here: what a simulation calculates for a later period
does not depend on which earlier periods it calculated first. Each case
compares against a fresh simulation that calculates only the later period.
The YEAR flow cases fail on 3.24.0 to 3.32.11. The integer, boolean and
STOCK year cases and the MONTH cases are controls: those variables never
cached a rescaled value in another unit, so they pass on those versions too.
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


class day_stock_input(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.DAY
    quantity_type = QuantityType.STOCK
    label = "Daily stock input; DAY variables have no set_input helper"


class month_stock_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    quantity_type = QuantityType.STOCK
    set_input = None
    label = "Monthly stock input with no set_input helper"


class year_stock_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    quantity_type = QuantityType.STOCK
    set_input = None
    label = "Yearly stock input with no set_input helper"


class year_flow_input_without_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = None
    label = "Yearly flow input with no set_input helper"


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
        day_stock_input,
        month_stock_input_without_helper,
        year_stock_input_without_helper,
        year_flow_input_without_helper,
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


# Inputs stored at a period in another unit than the definition period, as a
# variable with no set_input helper keeps them, and the later period asked.
OFF_UNIT_INPUTS = [
    ("day_stock_input", "2024-12", "2025-01-01"),
    ("month_stock_input_without_helper", str(BASE_YEAR), "2025-01"),
    ("year_stock_input_without_helper", MONTHS[-1], "2025"),
    ("year_flow_input_without_helper", MONTHS[-1], "2025"),
]


@pytest.mark.parametrize(
    "variable,input_period,later_period", OFF_UNIT_INPUTS, ids=lambda x: str(x)
)
def test_input_in_another_unit_still_carries_over(
    system, variable, input_period, later_period
):
    simulation = _simulation(system, {variable: {input_period: [24.0, 7.0]}})
    np.testing.assert_array_equal(
        simulation.calculate(variable, later_period), [24.0, 7.0]
    )


@pytest.mark.parametrize(
    "variable,input_period,later_period", OFF_UNIT_INPUTS, ids=lambda x: str(x)
)
def test_input_in_another_unit_set_on_a_branch_carries_over_there(
    system, variable, input_period, later_period
):
    simulation = _simulation(system, {})
    branch = simulation.get_branch("reform")
    branch.set_input(variable, input_period, [24.0, 7.0])
    np.testing.assert_array_equal(branch.calculate(variable, later_period), [24.0, 7.0])
    nested = branch.get_branch("nested")
    np.testing.assert_array_equal(nested.calculate(variable, later_period), [24.0, 7.0])


FLOW = "year_flow_input_without_helper"
FLOW_INPUT = {FLOW: {BASE_YEAR: [120.0, 12.0]}}


def test_derived_cache_ignores_a_same_named_branch_elsewhere(system):
    simulation = _simulation(system, FLOW_INPUT)
    nested = simulation.get_branch("a").get_branch("s")
    simulation.get_branch("s").set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    nested.calculate(FLOW, MONTHS[-1])
    np.testing.assert_array_equal(nested.calculate(FLOW, 2025), [120.0, 12.0])


def test_derived_cache_ignores_an_input_set_on_a_clone(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.calculate(FLOW, MONTHS[-1])
    clone = simulation.clone()
    clone.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])
    # The clone's year value wins over its month input.
    np.testing.assert_array_equal(clone.calculate(FLOW, 2025), [120.0, 12.0])


def test_derived_cache_ignores_a_parent_input_set_after_branching(system):
    simulation = _simulation(system, FLOW_INPUT)
    branch = simulation.get_branch("b")
    simulation.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    branch.calculate(FLOW, MONTHS[-1])
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [120.0, 12.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


def test_derived_cache_ignores_a_deleted_input_at_its_period(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    simulation.delete_arrays(FLOW, MONTHS[-1])
    simulation.calculate(FLOW, MONTHS[-1])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


def test_derivative_leaves_later_years_intact(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.derivative(FLOW, FLOW, MONTHS[-1])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


def test_own_unit_value_wins_over_an_input_at_another_unit(system):
    """A change from 3.24.0-3.32.11, which carried the later-starting month
    input; earlier versions carried the year value, as this does."""
    simulation = _simulation(system, FLOW_INPUT)
    simulation.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


def test_value_cached_only_at_another_unit_carries_over(system):
    simulation = _simulation(system, {})
    simulation.get_holder("day_stock_input").put_in_cache(
        np.array([24.0, 7.0]), periods.period("2024-12")
    )
    np.testing.assert_array_equal(
        simulation.calculate("day_stock_input", "2025-01-01"), [24.0, 7.0]
    )


# Lifecycle paths: nothing is tracked per period, so writes, deletions,
# replays and persistence cannot leave a rule stale.


def test_reform_replay_restores_an_input_at_another_unit(system):
    simulation = _simulation(system, {FLOW: {"2024-06": [600.0, 60.0]}})
    branch = simulation.get_branch("child")
    branch.calculate_divide(FLOW, "2024-06")
    branch.apply_reform({})
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [600.0, 60.0])


def test_dump_and_restore_keep_carrying_the_year_value(system, tmp_path):
    from policyengine_core.tools.simulation_dumper import (
        dump_simulation,
        restore_simulation,
    )

    simulation = _simulation(system, FLOW_INPUT)
    simulation.calculate_divide(FLOW, "2024-06")
    dump_simulation(simulation, str(tmp_path / "dump"))
    restored = restore_simulation(str(tmp_path / "dump"), system)
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])
    np.testing.assert_array_equal(restored.calculate(FLOW, 2025), [120.0, 12.0])


def test_deleting_a_branch_cache_exposes_the_ancestor_input(system):
    simulation = _simulation(system, {FLOW: {"2024-06": [600.0, 60.0]}})
    branch = simulation.get_branch("child")
    branch.calculate_divide(FLOW, "2024-06")
    branch.get_holder(FLOW).delete_arrays("2024-06", "child")
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [600.0, 60.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [600.0, 60.0])


def test_deleting_a_branch_input_does_not_expose_the_ancestor_twelfth(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.calculate_divide(FLOW, "2024-06")
    branch = simulation.get_branch("child")
    branch.set_input(FLOW, "2024-06", [600.0, 60.0])
    branch.get_holder(FLOW).delete_arrays("2024-06", "child")
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [120.0, 12.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])
