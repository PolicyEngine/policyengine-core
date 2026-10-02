"""ADD and DIVIDE results are cached only where a plain read returns them.

``calculate_add`` and ``calculate_divide`` used to store their result at the
requested period, where every later plain ``calculate`` of the variable at
that period found it. That is right for a FLOW variable, whose plain value
over a period of another unit is exactly that sum or twelfth, and wrong
elsewhere: a STOCK variable's plain value over a year is its last month's,
and over a month the year's. A plain read then returned 120 or 1 instead of
10 or 12, depending only on whether the option had run first.
"""

import pytest

from tests.fixtures.option_caches import (
    DAY,
    FLOW,
    MONTH,
    PROBE,
    STOCK,
    YEAR,
    build,
    make_probe,
    run,
)


def fresh_and_after(probe, inputs, target, prior, carry_over=False):
    """``target`` in a new simulation, and after running ``prior`` first."""
    fresh = build(probe, inputs, carry_over)
    used = build(probe, inputs, carry_over)
    prior_result = run(used, *prior)
    return run(fresh, *target), run(used, *target), prior_result


def test_add_over_a_monthly_stock_leaves_its_annual_value_alone():
    probe = make_probe(MONTH, STOCK)
    fresh, after, prior = fresh_and_after(
        probe,
        {"2012-01": 10},
        ("calculate", "2012"),
        ("add", "2012"),
        carry_over=True,
    )
    assert prior == [120.0]
    assert fresh == after == [10.0]


def test_divide_over_a_yearly_stock_leaves_its_monthly_value_alone():
    probe = make_probe(YEAR, STOCK, value_type=int)
    fresh, after, prior = fresh_and_after(
        probe,
        {"2012": 12},
        ("calculate", "2012-06"),
        ("divide", "2012-06"),
    )
    assert prior == [1.0]
    assert fresh == after == [12.0]


def test_stock_with_a_formula_keeps_its_last_month_value():
    probe = make_probe(MONTH, STOCK, with_formula=True)
    fresh, after, prior = fresh_and_after(
        probe, {}, ("calculate", "2012"), ("add", "2012")
    )
    assert prior == [sum(1200.0 + month for month in range(1, 13))]
    assert fresh == after == [1212.0]


def test_flow_add_is_still_cached_as_the_plain_annual_value():
    probe = make_probe(MONTH, FLOW, with_formula=True)
    simulation = build(probe)
    total = simulation.calculate_add(PROBE, "2012")
    holder = simulation.get_holder(PROBE)
    assert holder.get_array("2012").tolist() == total.tolist()
    assert simulation.calculate(PROBE, "2012").tolist() == total.tolist()


def test_flow_divide_is_still_cached_as_the_plain_monthly_value():
    probe = make_probe(YEAR, FLOW, with_formula=True)
    simulation = build(probe)
    twelfth = simulation.calculate_divide(PROBE, "2012-03")
    holder = simulation.get_holder(PROBE)
    assert holder.get_array("2012-03").tolist() == twelfth.tolist()
    assert twelfth.tolist() == [pytest.approx(1012.0 / 12)]


def test_add_does_not_replace_an_input_stored_at_the_year():
    # A monthly FLOW variable whose input is stored at the year itself.
    probe = make_probe(MONTH, FLOW, with_formula=True, set_input=None)
    fresh, after, prior = fresh_and_after(
        probe, {"2012": 7}, ("calculate", "2012"), ("add", "2012")
    )
    assert prior == [sum(1200.0 + month for month in range(1, 13))]
    assert fresh == after == [7.0]


def test_divide_does_not_replace_an_input_stored_at_the_month():
    probe = make_probe(YEAR, FLOW, with_formula=True, set_input=None)
    fresh, after, prior = fresh_and_after(
        probe, {"2012-02": 24}, ("calculate", "2012-02"), ("divide", "2012-02")
    )
    assert prior == [pytest.approx(1012.0 / 12)]
    assert fresh == after == [24.0]


@pytest.mark.parametrize(
    "definition_period, period",
    [(YEAR, "year:2012:2"), (MONTH, "month:2012-01:3")],
)
def test_add_over_several_own_periods_does_not_make_a_plain_read_succeed(
    definition_period, period
):
    probe = make_probe(definition_period, FLOW, with_formula=True)
    fresh, after, prior = fresh_and_after(
        probe, {}, ("calculate", period), ("add", period)
    )
    assert isinstance(prior, list)
    assert fresh == after == "ValueError"


def test_add_over_a_day_variable_leaves_its_monthly_value_alone():
    # A plain read of a day variable over a month runs its formula for the
    # month; it does not sum the days.
    probe = make_probe(DAY, FLOW, with_formula=True)
    fresh, after, prior = fresh_and_after(
        probe, {}, ("calculate", "2012-01"), ("add", "2012-01")
    )
    assert prior == [31 * 1201.0]
    assert fresh == after == [1201.0]


def test_stock_add_in_a_branch_leaves_the_branch_annual_value_alone():
    probe = make_probe(MONTH, STOCK, with_formula=True)
    fresh = build(probe).get_branch("other")
    used = build(probe).get_branch("other")
    used.calculate_add(PROBE, "2012")
    assert run(used, "calculate", "2012") == run(fresh, "calculate", "2012")
    assert run(used, "calculate", "2012") == [1212.0]


def test_a_twelfth_of_an_integer_flow_is_not_cached_truncated():
    # Storing 1012 / 12 in an integer variable would truncate it to 84, so a
    # second monthly read would differ from the first.
    probe = make_probe(YEAR, FLOW, value_type=int, with_formula=True)
    fresh, after, prior = fresh_and_after(
        probe, {}, ("calculate", "2012-01"), ("calculate", "2012-01")
    )
    assert fresh == after == prior == [pytest.approx(1012 / 12)]
