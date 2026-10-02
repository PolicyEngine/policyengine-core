"""Auto-carry-over and values cached at a unit other than the definition period.

A YEAR flow variable requested for a month caches a twelfth of the year's
value at that month (``calculate_divide``), and a sum over several
sub-periods is cached at the larger period (``calculate_add``). Since 3.24.0
auto-carry-over took the latest-starting known period of any unit, so once a
month of 2024 after January had been requested, a YEAR flow input known only
for 2024 carried a twelfth of its value into 2025.

policyengine-us computes ``monthly_age`` from ``age`` one month at a time.
A microsimulation that calculated 2024 and then 2025 on a single-year
dataset therefore aged every person to a twelfth of their age in 2025. On a
3,000-household Enhanced CPS subsample, 2025 federal income tax came to
$31.6bn, against $1,846.9bn in a simulation that calculated only 2025.

A variable with a ``set_input`` helper now prefers periods at its definition
unit. The helper stores inputs at the definition period, so a value at
another unit beside them is ordinarily a cache derived from them. A value at
another unit can still be a genuine input: one stored before a reform changed
the variable's definition period or helper, one restored from a simulation
with other definitions, or one written by a custom helper. So other units
are used when the variable has nothing at its definition unit. Variables
without a helper (falsy ``set_input``: DAY variables, ``set_input = None``)
keep the rule of 3.24.0 to 3.32.x, the latest period of any unit.

Known limits, pinned below: a helper variable that holds both a value at its
definition unit and a later genuine input at another unit carries the former;
a variable without a helper still carries a cached twelfth, as before.

The order invariant pinned for YEAR inputs and MONTH flow inputs: what a
simulation calculates for a later period does not depend on which earlier
periods it calculated first. Each such case compares against a fresh
simulation that calculates only the later period. The YEAR flow cases fail on
3.24.0 to 3.32.x; the integer, boolean and STOCK year cases and the MONTH
cases are controls that pass there too. (A MONTH stock input asked for a
later year and then an earlier month of that year still gets the default for
the month; that predates this change.)
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
from policyengine_core.holders import set_input_dispatch_by_period
from policyengine_core.reforms import Reform
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import QuantityType, Variable

BASE_YEAR = 2024
MONTHS = [f"{BASE_YEAR}-{month:02d}" for month in range(1, 13)]


# Variables with a set_input helper (the default for YEAR and MONTH).


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


class carried_monthly_stock(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.MONTH
    quantity_type = QuantityType.STOCK
    label = "Monthly stock input with no formula and no uprating"


class yearly_flow_with_dispatch_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = set_input_dispatch_by_period
    label = "Yearly flow input whose helper copies rather than divides"


class yearly_flow_with_false_helper(Variable):
    value_type = float
    entity = entities.Person
    definition_period = periods.YEAR
    set_input = False
    label = "Yearly flow input with a falsy set_input, treated as no helper"


# Variables without a helper.


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
FLOW = "carried_flow"
FLOW_INPUT = {FLOW: {BASE_YEAR: [120.0, 12.0]}}
NO_HELPER_FLOW = "year_flow_input_without_helper"
ALL_VARIABLES = (
    carried_flow,
    carried_stock,
    carried_count,
    carried_flag,
    carried_monthly_flow,
    carried_monthly_stock,
    yearly_flow_with_dispatch_helper,
    yearly_flow_with_false_helper,
    day_stock_input,
    month_stock_input_without_helper,
    year_stock_input_without_helper,
    year_flow_input_without_helper,
)


def _new_system():
    system = CountryTaxBenefitSystem()
    system.auto_carry_over_input_variables = True
    system.add_variables(*ALL_VARIABLES)
    return system


@pytest.fixture(scope="module")
def system():
    return _new_system()


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
    [str(BASE_YEAR)],
    [MONTHS[0]],
    [MONTHS[5]],
    [MONTHS[-1]],
    MONTHS,
    [MONTHS[-1], str(BASE_YEAR)],
    [str(BASE_YEAR), *MONTHS],
]
LATER_PERIODS = ["2025", "2027", "2025-03", "2027-12"]


def _ids(requests):
    return "+".join(requests)


# Variables with a helper: never carry a value at another unit.


def test_month_cache_of_year_input_does_not_carry_into_next_year(system):
    """The policyengine-us ``monthly_age`` sequence, on a synthetic variable."""
    simulation = _simulation(system, {FLOW: {BASE_YEAR: [40.0, 6.0]}})

    december = simulation.calculate(FLOW, MONTHS[-1])
    np.testing.assert_allclose(december, [40.0 / 12, 6.0 / 12])

    np.testing.assert_array_equal(
        simulation.calculate(FLOW, BASE_YEAR + 1), [40.0, 6.0]
    )


@pytest.mark.parametrize("later_year", [2025, 2027])
@pytest.mark.parametrize("variable", sorted(YEAR_INPUTS))
def test_year_input_carries_over_unchanged(system, variable, later_year):
    simulation = _simulation(system, {variable: {BASE_YEAR: YEAR_INPUTS[variable]}})
    np.testing.assert_array_equal(
        simulation.calculate(variable, later_year), YEAR_INPUTS[variable]
    )


@pytest.mark.parametrize("later_period", LATER_PERIODS)
@pytest.mark.parametrize("earlier", EARLIER_REQUESTS, ids=_ids)
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
        np.testing.assert_array_equal(result, YEAR_INPUTS[variable])


@pytest.mark.parametrize("later_period", LATER_PERIODS)
@pytest.mark.parametrize("earlier", EARLIER_REQUESTS, ids=_ids)
def test_year_input_branch_does_not_depend_on_earlier_requests(
    system, earlier, later_period
):
    """A branch forked after the earlier requests clones their month caches."""
    inputs = {FLOW: {BASE_YEAR: YEAR_INPUTS[FLOW]}}
    fresh = _simulation(system, inputs).calculate(FLOW, later_period)

    simulation = _simulation(system, inputs)
    for period in earlier:
        simulation.calculate(FLOW, period)
    branch = simulation.get_branch("itemizing")

    np.testing.assert_array_equal(branch.calculate(FLOW, later_period), fresh)
    np.testing.assert_array_equal(simulation.calculate(FLOW, later_period), fresh)


MONTHLY_INPUT_PATTERNS = {
    "every_month": {month: [float(i + 1), 100.0 + i] for i, month in enumerate(MONTHS)},
    "march_only": {MONTHS[2]: [3.0, 103.0]},
    "december_only": {MONTHS[-1]: [12.0, 112.0]},
}


@pytest.mark.parametrize("later_period", ["2025-01", "2025-07", "2026"])
@pytest.mark.parametrize("earlier", EARLIER_REQUESTS, ids=_ids)
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


def test_no_carry_over_backwards_from_a_later_input(system):
    simulation = _simulation(system, {FLOW: {2026: [120.0, 12.0]}})
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [0.0, 0.0])


def test_variable_with_helper_uses_another_unit_when_it_has_nothing_else(system):
    """As before: core cannot tell this value from a genuine input."""
    simulation = _simulation(system, {})
    simulation.get_holder(FLOW).put_in_cache(
        np.array([10.0, 1.0]), periods.period(MONTHS[-1])
    )
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [10.0, 1.0])


def test_stock_month_cache_from_an_explicit_divide_does_not_carry(system):
    simulation = _simulation(system, {"carried_stock": {BASE_YEAR: [40.0, 6.0]}})
    simulation.calculate_divide("carried_stock", MONTHS[-1])
    np.testing.assert_array_equal(
        simulation.calculate("carried_stock", 2025), [40.0, 6.0]
    )


def test_flow_with_a_dispatch_helper_does_not_carry_a_twelfth(system):
    variable = "yearly_flow_with_dispatch_helper"
    simulation = _simulation(system, {variable: {BASE_YEAR: [120.0, 12.0]}})
    simulation.calculate(variable, MONTHS[-1])
    np.testing.assert_array_equal(simulation.calculate(variable, 2025), [120.0, 12.0])


def test_month_cache_does_not_block_an_anchored_year_request(system):
    """The early default return also looks only at the preferred periods."""
    simulation = _simulation(system, FLOW_INPUT)
    simulation.calculate(FLOW, MONTHS[-1])
    np.testing.assert_array_equal(
        simulation.calculate(FLOW, "year:2024-06"), [120.0, 12.0]
    )


def test_branch_never_carries_an_inherited_twelfth(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.calculate(FLOW, "2024-06")
    branch = simulation.get_branch("child")
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [120.0, 12.0])


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
    # The helper stores a month input for a yearly variable as the year
    # starting that month, which then carries over in the clone.
    clone.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])
    np.testing.assert_array_equal(clone.calculate(FLOW, 2025), [999.0, 999.0])


def test_derived_cache_ignores_a_parent_input_set_after_branching(system):
    simulation = _simulation(system, FLOW_INPUT)
    branch = simulation.get_branch("b")
    simulation.set_input(FLOW, MONTHS[-1], [999.0, 999.0])
    branch.calculate(FLOW, MONTHS[-1])
    np.testing.assert_array_equal(branch.calculate(FLOW, 2025), [120.0, 12.0])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [999.0, 999.0])


def test_derivative_leaves_later_years_intact(system):
    simulation = _simulation(system, FLOW_INPUT)
    simulation.derivative(FLOW, FLOW, MONTHS[-1])
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


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


# Variables without a helper keep the rule of 3.24.0 to 3.32.x: the latest
# known period of any unit.

OFF_UNIT_INPUTS = [
    ("day_stock_input", "2024-12", "2025-01-01"),
    ("month_stock_input_without_helper", str(BASE_YEAR), "2025-01"),
    ("year_stock_input_without_helper", MONTHS[-1], "2025"),
    (NO_HELPER_FLOW, MONTHS[-1], "2025"),
]


@pytest.mark.parametrize(
    "variable,input_period,later_period", OFF_UNIT_INPUTS, ids=lambda x: str(x)
)
def test_input_in_another_unit_carries_over(
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


def test_latest_of_several_inputs_in_another_unit_carries_over(system):
    simulation = _simulation(
        system, {NO_HELPER_FLOW: {"2024-03": [3.0, 3.0], "2024-09": [9.0, 9.0]}}
    )
    np.testing.assert_array_equal(
        simulation.calculate(NO_HELPER_FLOW, 2025), [9.0, 9.0]
    )


def test_later_input_in_another_unit_beats_an_earlier_cached_default(system):
    """A default cached for 2025 does not shadow a later month input."""
    inputs = {NO_HELPER_FLOW: {"2025-09": [68.0, 6.0]}}
    fresh = _simulation(system, inputs).calculate(NO_HELPER_FLOW, 2026)

    simulation = _simulation(system, {})
    simulation.calculate(NO_HELPER_FLOW, 2025)
    simulation.set_input(NO_HELPER_FLOW, "2025-09", [68.0, 6.0])
    result = simulation.calculate(NO_HELPER_FLOW, 2026)

    np.testing.assert_array_equal(fresh, [68.0, 6.0])
    np.testing.assert_array_equal(result, fresh)


def test_input_in_another_unit_beats_an_earlier_year_sum(system):
    inputs = {NO_HELPER_FLOW: {"2025-03": [23.0, 2.0]}}
    fresh = _simulation(system, inputs).calculate(NO_HELPER_FLOW, 2026)

    simulation = _simulation(system, inputs)
    simulation.calculate_add(NO_HELPER_FLOW, BASE_YEAR)
    result = simulation.calculate(NO_HELPER_FLOW, 2026)

    np.testing.assert_array_equal(fresh, [23.0, 2.0])
    np.testing.assert_array_equal(result, fresh)


def test_reform_replay_restores_an_input_at_another_unit(system):
    simulation = _simulation(system, {NO_HELPER_FLOW: {"2024-06": [600.0, 60.0]}})
    branch = simulation.get_branch("child")
    branch.calculate_divide(NO_HELPER_FLOW, "2024-06")
    branch.apply_reform({})
    np.testing.assert_array_equal(branch.calculate(NO_HELPER_FLOW, 2025), [600.0, 60.0])


def test_deleting_a_branch_cache_exposes_the_ancestor_input(system):
    simulation = _simulation(system, {NO_HELPER_FLOW: {"2024-06": [600.0, 60.0]}})
    branch = simulation.get_branch("child")
    branch.calculate_divide(NO_HELPER_FLOW, "2024-06")
    branch.get_holder(NO_HELPER_FLOW).delete_arrays("2024-06", "child")
    np.testing.assert_array_equal(branch.calculate(NO_HELPER_FLOW, 2025), [600.0, 60.0])
    np.testing.assert_array_equal(
        simulation.calculate(NO_HELPER_FLOW, 2025), [600.0, 60.0]
    )


def test_variable_without_helper_still_carries_a_derived_twelfth(system):
    """A known limitation, unchanged here: such a variable can hold an input at
    any unit, so carry-over cannot tell a cached twelfth from an input and
    takes the latest period. No country package has such a variable."""
    simulation = _simulation(system, {NO_HELPER_FLOW: {BASE_YEAR: [120.0, 12.0]}})
    simulation.calculate(NO_HELPER_FLOW, MONTHS[-1])
    np.testing.assert_array_equal(
        simulation.calculate(NO_HELPER_FLOW, 2025), [10.0, 1.0]
    )


def test_falsy_set_input_is_treated_as_no_helper(system):
    variable = "yearly_flow_with_false_helper"
    simulation = _simulation(
        system,
        {variable: {BASE_YEAR: [120.0, 12.0], MONTHS[-1]: [999.0, 999.0]}},
    )
    np.testing.assert_array_equal(simulation.calculate(variable, 2025), [999.0, 999.0])


def test_helper_variable_with_both_units_carries_its_own_unit(system):
    """Known limit: a genuine later input at another unit loses to a value at
    the definition unit. Reached only when a value at another unit is a real
    input, which the built-in helpers never store."""
    simulation = _simulation(system, FLOW_INPUT)
    simulation.get_holder(FLOW).put_in_cache(
        np.array([999.0, 999.0]), periods.period(MONTHS[-1])
    )
    np.testing.assert_array_equal(simulation.calculate(FLOW, 2025), [120.0, 12.0])


# Supported paths that leave a helper variable with a genuine input at another
# unit. Each builds its own system: a reform must not touch the shared one.


def _yearly_stock_as_monthly():
    class carried_stock(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.MONTH
        quantity_type = QuantityType.STOCK
        label = "carried_stock, redefined monthly"

    return carried_stock


def _monthly_stock_as_yearly():
    class carried_monthly_stock(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        quantity_type = QuantityType.STOCK
        label = "carried_monthly_stock, redefined yearly"

    return carried_monthly_stock


class YearlyStockBecomesMonthly(Reform):
    def apply(self):
        self.update_variable(_yearly_stock_as_monthly())


class MonthlyStockBecomesYearly(Reform):
    def apply(self):
        self.update_variable(_monthly_stock_as_yearly())


class NoHelperStockGetsHelper(Reform):
    def apply(self):
        class year_stock_input_without_helper(Variable):
            value_type = float
            entity = entities.Person
            definition_period = periods.YEAR
            quantity_type = QuantityType.STOCK
            label = "Yearly stock input, now with the default helper"

        self.update_variable(year_stock_input_without_helper)


def test_input_kept_when_a_reform_makes_a_yearly_variable_monthly():
    simulation = _simulation(_new_system(), {"carried_stock": {BASE_YEAR: [40.0, 6.0]}})
    simulation.apply_reform(YearlyStockBecomesMonthly)
    np.testing.assert_array_equal(
        simulation.calculate("carried_stock", "2025-03"), [40.0, 6.0]
    )


def test_input_kept_when_a_reform_makes_a_monthly_variable_yearly():
    simulation = _simulation(
        _new_system(), {"carried_monthly_stock": {BASE_YEAR: [3.0, 4.0]}}
    )
    simulation.apply_reform(MonthlyStockBecomesYearly)
    np.testing.assert_array_equal(
        simulation.calculate("carried_monthly_stock", 2025), [3.0, 4.0]
    )


def test_input_kept_when_a_reform_gives_a_variable_a_helper():
    variable = "year_stock_input_without_helper"
    simulation = _simulation(_new_system(), {variable: {MONTHS[-1]: [40.0, 6.0]}})
    simulation.apply_reform(NoHelperStockGetsHelper)
    np.testing.assert_array_equal(simulation.calculate(variable, 2025), [40.0, 6.0])


def test_input_kept_when_restored_into_a_system_with_another_period(tmp_path):
    from policyengine_core.tools.simulation_dumper import (
        dump_simulation,
        restore_simulation,
    )

    simulation = _simulation(_new_system(), {"carried_stock": {BASE_YEAR: [40.0, 6.0]}})
    dump_simulation(simulation, str(tmp_path / "dump"))
    monthly_system = _new_system()
    monthly_system.update_variable(_yearly_stock_as_monthly())
    restored = restore_simulation(str(tmp_path / "dump"), monthly_system)
    np.testing.assert_array_equal(
        restored.calculate("carried_stock", "2025-03"), [40.0, 6.0]
    )
