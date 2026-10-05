"""Uprating depends only on the inputs, not on what was calculated first.

A variable with ``uprating`` and no formula result for a period takes the
latest earlier value stored in its own unit and multiplies it by the ratio of
the uprating index at the two period starts. Until this fix,
``Simulation._calculate`` took that value from any stored period, including
one the simulation had itself calculated, so calculating an intermediate
period first changed the result:

* An integer value truncated at each intermediate year compounded: an input of
  1001 for 2012 gave 1116 for 2015 asked alone, and 1115 after 2013 and 2014.
* A float32 value rounded at each intermediate year compounded the rounding.
* A value masked by ``defined_for``, or a default cached at an intermediate
  year, replaced the input from then on: someone ineligible in 2013 only got
  0 for 2015 after 2013 was calculated, and the input uprated otherwise.
* A default, an input stored in another unit and carried into an
  intermediate year, or the result of a formula a reform gave the variable up
  to an ``end`` became a value to uprate, where the period asked alone gave
  the default, the carried input or the uprated input.

Uprating now starts only from an input this branch reads
(``Holder.get_input_periods``): values the simulation calculated are marked
derived when cached and skipped, as auto-carry-over skips them, and so are
periods stored only under branches this one cannot read. The rule is in
``tests/fixtures/uprating_order.py``
and the order-independence property in ``test_uprating_order_property.py``.
Each regression here compares with a fresh simulation that calculates only
the period in question, byte for byte.

Two earlier inputs can also start on the same day: a yearly variable stores a
``year:2012:2`` input as given, beside one for ``2012`` (a ``set_input``
helper is only called for an input in another unit). The uprating source was
then whichever was stored first, or the one in memory over the one on disk.
It is now the one that ends last, the input auto-carry-over would carry.

A yearly input for a monthly variable is stored month by month by its
``set_input`` helper. The helpers took a month the simulation had calculated
for an input and kept it, so the uprating rule above skipped it. They now
replace it, and ``Simulation.set_input`` drops what ``calculate`` returned
for those months (the regressions at the end of this module).
"""

from __future__ import annotations

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import entities
from policyengine_core.experimental import MemoryConfig
from policyengine_core.reforms import Reform
from policyengine_core.variables import Variable
from tests.fixtures.uprating_order import (
    MONTHLY_INDEX,
    UPRATING,
    alone,
    assert_bitwise_equal,
    build_system,
    can_store_on_disk,
    index,
    reference,
    simulation,
)

SYSTEMS = {True: build_system(True), False: build_system(False)}


@pytest.fixture(params=[True, False], ids=["carry_over", "no_carry_over"])
def system(request):
    return SYSTEMS[request.param]


def _after(system, inputs, requests, variable, period):
    built = simulation(system, inputs)
    for requested_variable, requested_period in requests:
        built.calculate(requested_variable, requested_period)
    return built.calculate(variable, period)


def test_integer_truncation_does_not_compound(system):
    inputs = {"uprated_count": {"2012": [1001, 77]}}
    requests = [("uprated_count", "2013"), ("uprated_count", "2014")]
    result = _after(system, inputs, requests, "uprated_count", "2015")
    # Before the fix: [1115, 83], truncated at 2013 and again at 2014.
    np.testing.assert_array_equal(result, [1116, 85])
    assert_bitwise_equal(result, alone(system, inputs, "uprated_count", "2015"))


def test_float32_rounding_does_not_compound(system):
    inputs = {"uprated": {"2012": [1001.3, 77.7]}}
    requests = [("uprated", "2013"), ("uprated", "2014")]
    result = _after(system, inputs, requests, "uprated", "2015")
    assert_bitwise_equal(result, alone(system, inputs, "uprated", "2015"))
    np.testing.assert_allclose(
        result, np.array([1001.3, 77.7]) * index(2015) / index(2012), rtol=1e-6
    )


def test_value_masked_by_defined_for_is_not_uprated(system):
    inputs = {
        "uprated_if_eligible": {"2012": [1000, 1000]},
        "eligible": {
            "2012": [True, True],
            "2013": [False, True],
            "2014": [True, True],
            "2015": [True, True],
        },
    }
    requests = [("uprated_if_eligible", "2013"), ("uprated_if_eligible", "2014")]
    result = _after(system, inputs, requests, "uprated_if_eligible", "2015")
    # Before the fix: [0, 1115.16]; the 2013 mask carried into 2015.
    np.testing.assert_allclose(
        result, [1000 * index(2015) / index(2012)] * 2, rtol=1e-6
    )
    assert_bitwise_equal(result, alone(system, inputs, "uprated_if_eligible", "2015"))


def test_default_cached_where_defined_for_is_false_is_not_uprated(system):
    inputs = {
        "uprated_if_eligible": {"2012": [1000, 1000]},
        "eligible": {
            "2012": [True, True],
            "2013": [False, False],
            "2014": [True, True],
        },
    }
    requests = [("uprated_if_eligible", "2013")]
    result = _after(system, inputs, requests, "uprated_if_eligible", "2014")
    # Before the fix: [0, 0], the default cached for 2013.
    np.testing.assert_allclose(
        result, [1000 * index(2014) / index(2012)] * 2, rtol=1e-6
    )
    assert_bitwise_equal(result, alone(system, inputs, "uprated_if_eligible", "2014"))


def test_cached_default_value_is_not_uprated(system):
    requests = [("uprated_with_default", "2013")]
    result = _after(system, {}, requests, "uprated_with_default", "2015")
    # Before the fix: 10 * index(2015) / index(2013).
    np.testing.assert_array_equal(result, [10, 10])
    assert_bitwise_equal(result, alone(system, {}, "uprated_with_default", "2015"))


def test_input_in_another_unit_is_not_uprated_through_a_calculated_year(system):
    inputs = {"uprated_any_unit": {"2012-03": [10, 20]}}
    requests = [("uprated_any_unit", "2013")]
    result = _after(system, inputs, requests, "uprated_any_unit", "2015")
    # Asked alone, 2015 has no earlier yearly input to uprate: auto-carry-over
    # carries the monthly input unchanged, or it gets the default. Before the
    # fix, calculating 2013 first cached that value at 2013, and 2015 uprated
    # it.
    expected = [10, 20] if system.auto_carry_over_input_variables else [0, 0]
    np.testing.assert_array_equal(result, expected)
    assert_bitwise_equal(result, alone(system, inputs, "uprated_any_unit", "2015"))


def test_reform_formula_result_is_not_uprated_past_the_formula_end(system):
    """Core rejects a variable declaring both a formula and ``uprating``, but
    a reform can give an uprated variable a formula, which keeps the
    inherited ``uprating``. Past the formula's ``end`` the variable is uprated
    again, from its input, not from a formula result."""

    class formula_until_2013(Reform):
        def apply(self):
            class uprated(Variable):
                value_type = float
                entity = entities.Person
                definition_period = periods.YEAR
                end = "2013-12-31"
                label = "Uprated yearly input given a formula up to 2013"

                def formula(person, period, parameters):
                    return person.filled_array(42)

            self.update_variable(uprated)

    reformed = formula_until_2013(system)
    assert reformed.get_variable("uprated").uprating == UPRATING
    inputs = {"uprated": {"2012": [1, 2]}}
    result = _after(reformed, inputs, [("uprated", "2013")], "uprated", "2015")
    # Before the fix: the 2013 formula result, 42, uprated to 2015.
    np.testing.assert_allclose(
        result, np.array([1, 2]) * index(2015) / index(2012), rtol=1e-6
    )
    assert_bitwise_equal(result, alone(reformed, inputs, "uprated", "2015"))


def test_monthly_rounding_does_not_compound(system):
    inputs = {"uprated_monthly": {"2011-06": [1001.3, 77.7]}}
    requests = [("uprated_monthly", f"2012-{month:02d}") for month in range(1, 13)]
    requests += [("uprated_monthly", "2013-01")]
    result = _after(system, inputs, requests, "uprated_monthly", "2014-03")
    assert_bitwise_equal(result, alone(system, inputs, "uprated_monthly", "2014-03"))
    np.testing.assert_allclose(
        result, np.array([1001.3, 77.7]) * index(2014) / index(2011), rtol=1e-6
    )


@pytest.mark.parametrize("nested", [False, True])
def test_branch_forked_after_intermediate_years_uprates_the_input(system, nested):
    inputs = {"uprated_count": {"2012": [1001, 77]}}
    built = simulation(system, inputs)
    built.calculate("uprated_count", "2013")
    built.calculate("uprated_count", "2014")
    branch = built.get_branch("reform")
    if nested:
        branch = branch.get_branch("nested")
    result = branch.calculate("uprated_count", "2015")
    np.testing.assert_array_equal(result, [1116, 85])
    assert_bitwise_equal(result, alone(system, inputs, "uprated_count", "2015"))


def test_parent_calculation_does_not_hide_a_branch_input():
    """An input stored only under a branch keeps its provenance when the
    parent calculates a value for the same period."""
    system = SYSTEMS[True]
    built = simulation(system, {"uprated": {"2011": [1, 2]}})
    built.get_holder("uprated").set_input(
        periods.period("2012"), np.array([300.0, 400.0]), "other"
    )
    built.calculate("uprated", "2012")
    branch = built.get_branch("other")
    np.testing.assert_allclose(
        branch.calculate("uprated", "2014"),
        np.array([300, 400]) * index(2014) / index(2012),
        rtol=1e-6,
    )


def test_period_stored_only_under_an_unreadable_branch_is_not_uprated(system):
    """A holder can store a period under a branch this simulation cannot
    read (here through ``Holder.set_input`` with another branch name). It is
    not an input this simulation can uprate from."""
    inputs = {"uprated_count": {"2012": [1001, 77]}}
    built = simulation(system, inputs)
    built.get_holder("uprated_count").set_input(
        periods.period("2013"), np.array([2001, 177]), "sibling"
    )
    # Before the fix: TypeError, the sibling's 2013 read back as None here.
    result = built.calculate("uprated_count", "2015")
    np.testing.assert_array_equal(result, [1116, 85])
    assert_bitwise_equal(result, alone(system, inputs, "uprated_count", "2015"))


# Guards against over-correction: inputs at intermediate periods still count.


def test_latest_earlier_input_is_uprated(system):
    inputs = {"uprated": {"2012": [1, 2], "2014": [300, 400]}}
    for requests in (
        [],
        [("uprated", "2013")],
        [("uprated", "2013"), ("uprated", "2016")],
    ):
        result = _after(system, inputs, requests, "uprated", "2015")
        np.testing.assert_allclose(
            result, np.array([300, 400]) * index(2015) / index(2014), rtol=1e-6
        )


def test_input_set_over_a_calculated_period_is_uprated(system):
    built = simulation(system, {"uprated": {"2012": [1, 2]}})
    built.calculate("uprated", "2013")
    built.set_input("uprated", "2013", np.array([300.0, 400.0]))
    np.testing.assert_allclose(
        built.calculate("uprated", "2015"),
        np.array([300, 400]) * index(2015) / index(2013),
        rtol=1e-6,
    )


def test_branch_input_over_a_period_the_parent_calculated_is_uprated(system):
    built = simulation(system, {"uprated": {"2012": [1, 2]}})
    built.calculate("uprated", "2013")
    branch = built.get_branch("reform")
    branch.set_input("uprated", "2013", np.array([300.0, 400.0]))
    np.testing.assert_allclose(
        branch.calculate("uprated", "2015"),
        np.array([300, 400]) * index(2015) / index(2013),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        built.calculate("uprated", "2015"),
        np.array([1, 2]) * index(2015) / index(2012),
        rtol=1e-6,
    )


def test_add_over_an_input_period_keeps_it_an_input(system):
    built = simulation(system, {"uprated_count": {"2012": [1001, 77]}})
    built.calculate_add("uprated_count", "2012")
    built.calculate("uprated_count", "2013")
    np.testing.assert_array_equal(built.calculate("uprated_count", "2015"), [1116, 85])


def test_uprated_value_is_cached_as_derived(system):
    built = simulation(system, {"uprated": {"2012": [1, 2]}})
    built.calculate("uprated", "2013")
    holder = built.get_holder("uprated")
    assert holder.is_derived(periods.period("2013"))
    assert not holder.is_derived(periods.period("2012"))


# Inputs that start on the same day.

TIED = {"year:2012:2": [1.0, 2.0], "2012": [5.0, 6.0]}


def _in_order(values, longer_first):
    order = list(values) if longer_first else list(values)[::-1]
    return {key: values[key] for key in order}


@pytest.mark.parametrize("variable", ["uprated_any_unit", "uprated"])
@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_inputs_starting_together_uprate_the_one_ending_last(
    system, variable, longer_first
):
    """``uprated`` has a ``set_input`` helper and ``uprated_any_unit`` none:
    both store ``year:2012:2`` as given."""
    inputs = {variable: _in_order(TIED, longer_first)}
    result = alone(system, inputs, variable, "2015")
    # Before the fix: [5, 6] uprated whenever 2012 was stored first.
    np.testing.assert_allclose(
        result, np.array([1, 2]) * index(2015) / index(2012), rtol=1e-6
    )
    assert_bitwise_equal(result, reference(system, inputs, variable, "2015"))


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_inputs_starting_together_uprate_the_longest_not_the_first_by_name(
    system, reverse
):
    """``year:2012:10`` ends last; as a string it sorts between ``2012`` and
    ``year:2012:2``."""
    values = {"2012": [5.0, 6.0], "year:2012:10": [1.0, 2.0], "year:2012:2": [3, 4]}
    order = list(values)[::-1] if reverse else list(values)
    inputs = {"uprated_any_unit": {key: values[key] for key in order}}
    result = alone(system, inputs, "uprated_any_unit", "2015")
    np.testing.assert_allclose(
        result, np.array([1, 2]) * index(2015) / index(2012), rtol=1e-6
    )
    assert_bitwise_equal(result, reference(system, inputs, "uprated_any_unit", "2015"))


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_monthly_inputs_starting_together_uprate_the_one_ending_last(
    system, longer_first
):
    values = {"month:2012-01:3": [1.0, 2.0], "2012-01": [5.0, 6.0]}
    inputs = {"uprated_monthly": _in_order(values, longer_first)}
    result = alone(system, inputs, "uprated_monthly", "2014-03")
    np.testing.assert_allclose(
        result, np.array([1, 2]) * index(2014) / index(2012), rtol=1e-6
    )
    assert_bitwise_equal(
        result, reference(system, inputs, "uprated_monthly", "2014-03")
    )


@pytest.mark.filterwarnings(
    "ignore::policyengine_core.warnings.memory_config_warning.MemoryConfigWarning"
)
@pytest.mark.parametrize(
    "longer_on_disk",
    [
        pytest.param(
            True,
            id="longer",
            marks=pytest.mark.skipif(
                not can_store_on_disk("year:2012:2"),
                reason="OnDiskStorage cannot store year:2012:2 on Windows (#526)",
            ),
        ),
        pytest.param(False, id="shorter"),
    ],
)
def test_inputs_starting_together_uprate_the_same_one_from_memory_or_disk(
    system, longer_on_disk
):
    """``get_known_periods`` lists the periods in memory before those on
    disk, whatever order they were stored in."""
    built = simulation(system, {})
    built.memory_config = MemoryConfig(max_memory_occupation=0)
    holder = built.get_holder("uprated_any_unit")
    holder._disk_storage = holder.create_disk_storage()
    on_disk, in_memory = ("year:2012:2", "2012")[:: 1 if longer_on_disk else -1]
    holder._on_disk_storable = False
    built.set_input("uprated_any_unit", in_memory, np.array(TIED[in_memory]))
    holder._on_disk_storable = True
    built.set_input("uprated_any_unit", on_disk, np.array(TIED[on_disk]))
    assert holder._disk_storage.has(periods.period(on_disk))
    assert holder._memory_storage.has(periods.period(in_memory))
    result = built.calculate("uprated_any_unit", "2015")
    expected = alone(system, {"uprated_any_unit": TIED}, "uprated_any_unit", "2015")
    assert_bitwise_equal(result, expected)
    np.testing.assert_allclose(
        result, np.array([1, 2]) * index(2015) / index(2012), rtol=1e-6
    )


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_branch_input_starting_with_a_parent_input_is_uprated_by_the_same_rule(
    system, longer_first
):
    first, second = ("year:2012:2", "2012")[:: 1 if longer_first else -1]
    built = simulation(system, {"uprated": {first: TIED[first]}})
    branch = built.get_branch("reform")
    branch.set_input("uprated", second, np.array(TIED[second]))
    np.testing.assert_allclose(
        branch.calculate("uprated", "2015"),
        np.array([1, 2]) * index(2015) / index(2012),
        rtol=1e-6,
    )


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_flat_uprating_reads_the_input_auto_carry_over_reads(longer_first):
    """Uprating by an index that never moves returns what auto-carry-over
    returns for the same inputs, ties included, as long as no input in the
    variable's own unit starts on the period's first day (see the next
    test)."""
    system = SYSTEMS[True]
    values = {**TIED, "2012-06": [7.0, 8.0], "2014-02": [9.0, 10.0]}
    ordered = _in_order(values, longer_first)
    built = simulation(system, {"uprated_flat": ordered, "carried_any_unit": ordered})
    for year in ("2013", "2014", "2015"):
        assert_bitwise_equal(
            built.calculate("uprated_flat", year),
            built.calculate("carried_any_unit", year),
            year,
        )
    np.testing.assert_array_equal(built.calculate("uprated_flat", "2015"), [1, 2])


def test_longer_input_starting_on_the_period_is_not_an_uprating_source(system):
    """The rule this change keeps: only inputs that start before the period
    are uprated from, so ``year:2012:2`` is not a source for 2012 itself.
    2012 gets 2011's input uprated; auto-carry-over would carry
    ``year:2012:2``. Without an earlier input, 2012 gets what auto-carry-over
    gives it. Whether that should change is PolicyEngine/policyengine-core#583."""
    inputs = {"uprated_any_unit": {"2011": [1.0, 2.0], "year:2012:2": [5.0, 6.0]}}
    np.testing.assert_allclose(
        alone(system, inputs, "uprated_any_unit", "2012"),
        np.array([1, 2]) * index(2012) / index(2011),
        rtol=1e-6,
    )
    inputs = {"uprated_any_unit": {"year:2012:2": [5.0, 6.0]}}
    expected = [5, 6] if system.auto_carry_over_input_variables else [0, 0]
    np.testing.assert_array_equal(
        alone(system, inputs, "uprated_any_unit", "2012"), expected
    )


@pytest.mark.parametrize(
    "stored,asked",
    [("day:9999-12-30:3", "9999-12-31"), ("day:2012-01-01:4000000", "2013-01-01")],
)
def test_an_input_ending_after_the_last_date_is_uprated(system, stored, asked):
    """A period that ends after 9999-12-31 has no ``stop`` (it raises
    ``OverflowError``); it is still an uprating source."""
    inputs = {"uprated_daily_flat": {stored: [10.0, 20.0]}}
    np.testing.assert_array_equal(
        alone(system, inputs, "uprated_daily_flat", asked), [10, 20]
    )
    assert_bitwise_equal(
        alone(system, inputs, "uprated_daily_flat", asked),
        reference(system, inputs, "uprated_daily_flat", asked),
    )


@pytest.mark.parametrize("longer_first", [True, False], ids=["longer", "shorter"])
def test_an_input_ending_after_the_last_date_ends_last(system, longer_first):
    values = {"day:2012-01-01:4000000": [10.0, 20.0], "day:2012-01-01:2": [1.0, 2.0]}
    inputs = {"uprated_daily_flat": _in_order(values, longer_first)}
    np.testing.assert_array_equal(
        alone(system, inputs, "uprated_daily_flat", "2013-01-01"), [10, 20]
    )


# A yearly input for a monthly variable goes through its ``set_input`` helper,
# which stores it in each month of the year: copied to each month for a stock
# (``set_input_dispatch_by_period``), divided between them for a flow
# (``set_input_divide_by_period``). A month the simulation calculated before
# the input was set is not an input, so the helper replaces it as it fills a
# month nothing is stored for. Before the fix, the helpers treated it as one:
# dispatch kept it and copied it into the later months as an input, and divide
# took it out of the year's total. The month also read back as calculated, and
# a later month was uprated from an earlier input month, or from the copies.

HELPER_INPUTS = {
    "uprated_monthly_stock": ([7.0, 9.0], [7.0, 9.0]),
    "uprated_monthly": ([84.0, 108.0], [7.0, 9.0]),
}
CALCULATED_FIRST = [["2012-12"], ["2012-06"], ["2012-06", "2012-12"], ["2012-01"]]


def _on(built, branch):
    if branch in ("reform", "nested"):
        built = built.get_branch("reform")
    if branch == "nested":
        built = built.get_branch("nested")
    return built


@pytest.mark.parametrize("branch", [None, "reform", "nested"])
@pytest.mark.parametrize("calculated", CALCULATED_FIRST, ids="+".join)
@pytest.mark.parametrize("variable", list(HELPER_INPUTS), ids=["dispatch", "divide"])
def test_helper_input_replaces_months_calculated_before_it(
    system, variable, calculated, branch
):
    yearly, monthly = HELPER_INPUTS[variable]
    built = _on(simulation(system, {}), branch)
    for month in calculated:
        built.calculate(variable, month)
    built.set_input(variable, "2012", np.array(yearly))

    fresh = _on(simulation(system, {}), branch)
    fresh.set_input(variable, "2012", np.array(yearly))
    holder = built.get_holder(variable)
    for month in ("2012-01", "2012-06", "2012-12"):
        result = built.calculate(variable, month)
        np.testing.assert_array_equal(result, monthly)
        assert_bitwise_equal(result, fresh.calculate(variable, month), month)
        assert not holder.is_derived(periods.period(month), built.branch_name)
    for month in ("2013-01", "2014-03"):
        assert_bitwise_equal(
            built.calculate(variable, month), fresh.calculate(variable, month), month
        )


@pytest.mark.parametrize("variable", list(HELPER_INPUTS), ids=["dispatch", "divide"])
def test_helper_input_is_uprated_from_its_last_month(system, variable):
    """The month after the year is uprated from December's input, not from an
    earlier month the helper filled while December held a calculated value."""
    if variable == "uprated_monthly_stock":
        ratio = MONTHLY_INDEX["2013-01-01"] / MONTHLY_INDEX["2012-12-01"]
    else:
        ratio = index(2013) / index(2012)
    yearly, monthly = HELPER_INPUTS[variable]
    built = simulation(system, {})
    built.calculate(variable, "2012-12")
    built.set_input(variable, "2012", np.array(yearly))
    np.testing.assert_allclose(
        built.calculate(variable, "2013-01"), np.array(monthly) * ratio, rtol=1e-6
    )


@pytest.mark.parametrize(
    "variable, expected",
    [
        # Dispatch copies an input it meets into the months after it.
        ("uprated_monthly_stock", {"2012-02": [7, 9], "2012-06": [1, 2]}),
        # Divide shares what the inputs it meets leave of the total.
        (
            "uprated_monthly",
            {"2012-02": [83 / 11, 106 / 11], "2012-06": [83 / 11, 106 / 11]},
        ),
    ],
    ids=["dispatch", "divide"],
)
def test_helper_input_keeps_an_input_already_set_in_its_year(
    system, variable, expected
):
    """Guard against over-correction: an input for a month of the year is
    still kept, whether or not other months were calculated."""
    yearly, _ = HELPER_INPUTS[variable]
    for calculated in ([], ["2012-06"], ["2012-03", "2012-06"]):
        built = simulation(system, {variable: {"2012-03": [1.0, 2.0]}})
        for month in calculated:
            built.calculate(variable, month)
        built.set_input(variable, "2012", np.array(yearly))
        np.testing.assert_array_equal(built.calculate(variable, "2012-03"), [1, 2])
        for month, values in expected.items():
            np.testing.assert_allclose(
                built.calculate(variable, month), values, rtol=1e-6
            )


class _no_change(Reform):
    def apply(self):
        pass


@pytest.mark.filterwarnings("ignore:Memory configuration is a feature")
@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("variable", list(HELPER_INPUTS), ids=["dispatch", "divide"])
def test_an_input_set_on_a_clone_is_not_replayed_by_its_source(
    system, variable, on_disk
):
    """``apply_reform`` keeps a simulation's inputs and drops what it
    calculated. A clone recorded its inputs in its source's record, so once
    a helper input on the clone replaced a month the source had calculated,
    the source's reform kept its calculated month as an input, and uprated
    from it."""
    inputs = {variable: {"2011-06": [5.0, 6.0]}}
    source = simulation(system, {})
    if on_disk:
        source.memory_config = MemoryConfig(max_memory_occupation=0)
        holder = source.get_holder(variable)
        holder._disk_storage = holder.create_disk_storage()
        holder._on_disk_storable = True
    source.set_input(variable, "2011-06", np.array(inputs[variable]["2011-06"]))
    source.calculate(variable, "2012-12")
    clone = source.clone()
    yearly, monthly = HELPER_INPUTS[variable]
    clone.set_input(variable, "2012", np.array(yearly))

    source.apply_reform(_no_change)
    clone.apply_reform(_no_change)

    december = periods.period("2012-12")
    assert (variable, "default", december) not in source._user_input_keys
    assert source.get_holder(variable).get_array(december) is None
    assert_bitwise_equal(
        source.calculate(variable, "2013-01"),
        alone(system, inputs, variable, "2013-01"),
    )
    np.testing.assert_array_equal(clone.calculate(variable, "2012-12"), monthly)
